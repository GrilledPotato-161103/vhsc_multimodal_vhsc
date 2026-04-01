import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class RGB_Action(nn.Module):
    def __init__(self, output_size, video_length, keep_time=False):
        super(RGB_Action, self).__init__()
        self.keep_time = keep_time
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-4])
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        if not self.keep_time:
            self.fc_block = nn.Sequential(
                nn.Linear(int(video_length/2)*64*14*14, 512), # 64C,video_len/2=T,28H,28W
                nn.ReLU(),
                nn.Linear(512, output_size)
            )
        else:
            # self.fc_block = nn.Sequential(
            #     nn.Linear(64*14*14, output_size//4), # 64C,video_len/2=T,28H,28W
            #     nn.ReLU(),
            #     nn.Linear(output_size//4, output_size//2),
            #     nn.ReLU(),
            #     nn.Linear(output_size//2, output_size)
            # )
            self.fc_block = nn.Sequential(
                nn.Linear(64*14*14, output_size//2), # 64C,video_len/2=T,28H,28W
                nn.BatchNorm1d(output_size//2),
                nn.ReLU(),
                nn.Linear(output_size//2, output_size)
             )
            #initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
         # Apply ResNet to each frame
        b, t, c, h, w = x.shape
        # print("Initial shape:",x.shape) #[8,30,3,224,224]
        x = x.view(-1, c, h, w)
        x = self.feature_extractor(x)
        post_fe_x = x.clone()
        # print("Post feature extractor shape:",x.shape) #[240,128,28,28] = [b*t,c,h,w] or [8,30,128,28,28]= [b,t,c,h,w]

        #insert time dim after b
        x = x.view(b, t, *x.shape[1:]) # [8,30,128,14,14]

        x = x.permute(0,2,1,3,4) # permute to BCTHW bc conv3d expects C before T

        # Apply 3D convolutional layers
        x = self.conv3d_block(x)
        # print("after conv3d",x.shape) # [8, 64, 15, 14, 14] #time dimension is 15 here too! it's just video_len/2
        b,c,t,h,w = x.shape

        x = x.view(b,-1) # flatten preserve batch_size
        # print(x.shape) #    [8, 188160]

        #residual connection
        post_fe_x = post_fe_x.view(b,-1).chunk(16,dim=1) #Note that post feature extractor is 2x bigger in the last 4 dims compared to post 3dconv 2^4=16 
        # print("post_fe_x shape",len(post_fe_x),post_fe_x[0].shape)
        chunks = torch.stack(post_fe_x,dim=1)
        # print("chunks shape",chunks.shape)
        summed_chunks = chunks.sum(dim=1)
        # print("summed chunks shape",summed_chunks.shape)
        x = x+summed_chunks #maybe this will help with gradient propogation
        # exit()

        if not self.keep_time:
            #Fully connected for class prediction
            x = self.fc_block(x)
        else:
            x = x.view(b,c,t,h,w) #reshape back to BCTHW
            x = x.permute(0,2,1,3,4) # permute back to BTCHW
            b,t_new,c,h,w = x.shape
            x = x.reshape(b*t_new,-1) #conv3d will decrease the original time by half, hence we call it t_new, technically h and w are also new lol
            # now pass each frame through the fc block
            x = self.fc_block(x)
            x = x.reshape(b,t_new,-1) #reshape back to BT, embeddings
            # The final output will be a batch of t/2 out_putsize vectors
        return x
    
#Define 1D CNN model
class IMU_CNN(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, imu_length, keep_time=False):
        super(IMU_CNN, self).__init__()
        self.hidden_size = hidden_size
        self.keep_time = keep_time
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_size//2, hidden_size//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # assuming starting dim is 6x180 output should be hidden_size/4 x 15 time decreases by factor of 12
        # Note in our cross-modal fusion we make hidden_size = hidden_size*2, and output_size = hidden_size, just look at when we instatiate this.

        if not self.keep_time:
            # Flatten and apply to all frames at once
            self.fc = nn.Sequential(
                nn.Linear(hidden_size//4*imu_length//12, output_size, dtype=torch.float32),
            )
        else:
            # Apply fc to every frame individually
            self.fc = nn.Sequential(
                nn.Linear(hidden_size//4, output_size, dtype=torch.float32),
            )

    def forward(self, x):
        x = x.permute(0,2,1) # permute to bs x channels x timesteps
        out = self.layers(x)

        if not self.keep_time:
            # Flatten and apply to all frames at once
            out = out.view(out.shape[0], -1)
            out = self.fc(out)
        else:
            # Apply fc to every frame individually
            out = out.permute(0,2,1) # permute back to bs x timesteps x channels
            b,t,c = out.shape
            out = out.reshape(b*t, -1)
            out = self.fc(out)
            out = out.reshape(b,t,-1)
            # the final output will be batch of t=15 output_sized vectors

        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.float32),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, dtype=torch.float32),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.layers(x)
        return out

class IMU_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IMU_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            MLP(input_size, hidden_size, int(hidden_size/4)),
            nn.Dropout(0.5),
            MLP(int(hidden_size/4), int(hidden_size/4), int(hidden_size/16)),
            nn.Linear(int(hidden_size/16), output_size, dtype=torch.float32)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.layers(x)
        return out

#Attnetion based feature fusion
class HAMLET(nn.Module):
    def __init__(self, cfg):
        super(HAMLET, self).__init__()
        self.hidden_size = cfg.hidden_size #here hiddent size will be the size the two features join at (addition)
        
        #Unimodal encoders
        self.rgb_model = RGB_Action(cfg.hidden_size, cfg.rgb_video_length)
        self.watch_accel_model = IMU_CNN(input_channels=cfg.num_imu_channels, hidden_size=cfg.hidden_size*2, output_size=cfg.hidden_size, imu_length=cfg.imu_length)
        # self.phone_accel_model = IMU_CNN(input_channels=cfg.num_imu_channels, hidden_size=cfg.hidden_size*2, output_size=cfg.hidden_size, imu_length=cfg.imu_length)
        self.phone_gyro_model = IMU_CNN(input_channels=cfg.num_imu_channels, hidden_size=cfg.hidden_size*2, output_size=cfg.hidden_size, imu_length=cfg.imu_length)
        # self.phone_orientation_model = IMU_CNN(input_channels=cfg.num_imu_channels, hidden_size=cfg.hidden_size*2, output_size=cfg.hidden_size, imu_length=cfg.imu_length)

        # Multimodal mutli-head self attention
        self.attn = nn.MultiheadAttention(cfg.hidden_size, num_heads=2, dropout=0, batch_first=True)

        # feed forward network
        self.joint_processing = IMU_MLP(cfg.hidden_size, cfg.hidden_size//2, cfg.num_classes)

    def forward(self, x, save_embedding=False):
        z_rgb = self.rgb_model(x[0])
        z_watch_accel = self.watch_accel_model(x[1][:, :, :3])
        # z_phone_accel = self.phone_accel_model(x[1][:, :, 3:6])
        z_phone_gyro = self.phone_gyro_model(x[1][:, :, 3:6])

        if save_embedding:
            return z_rgb, z_watch_accel, z_phone_gyro
        # z_phone_orientation = self.phone_orientation_model(x[1][:, :, 9:])
        # batch, features
        z_rgb = z_rgb.unsqueeze(1) # add time dimension
        z_watch_accel = z_watch_accel.unsqueeze(1)
        # z_phone_accel = z_phone_accel.unsqueeze(1)
        z_phone_gyro = z_phone_gyro.unsqueeze(1)
        # z_phone_orientation = z_phone_orientation.unsqueeze(1)
        # z_imu = torch.cat((z_watch_accel, z_phone_accel, z_phone_gyro, z_phone_orientation), dim=1)
        z_imu = torch.cat((z_watch_accel, z_phone_gyro), dim=1)

        z_features = torch.cat((z_rgb, z_imu), dim=1) # 
        out, _ = self.attn(z_features, z_features, z_features)
        # batch, seq, features = out.shape
        out = out.sum(dim=1)
        out = self.joint_processing(out)
        
        return out, None, None, None, None

class Reconstructor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Reconstructor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)
        # return F.normalize(self.net(x), p=2, dim=-1)
    
class ReconstructorSigma2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReconstructorSigma2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, x, rec):
        x = torch.cat((x, rec), dim=-1)
        x = self.net(x)
        x = F.softplus(x)
        # x = x / torch.norm(x, p=2, dim=-1, keepdim=True) # normalize the output to have unit norm
        return x
    
class OutputSigma2(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes):
        super(OutputSigma2, self).__init__()
        self.net_1 = nn.Sequential(
            nn.Linear(in_dim + num_classes, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim)
        )

    def forward(self, x, rec):
        x_ = self.net_1(torch.cat((x, rec), dim=-1)) + x
        out = self.net_2(x_)
        out = F.softplus(out)

        return out

class EmbeddingHAMLET(nn.Module):
    def __init__(self, cfg):
        super(EmbeddingHAMLET, self).__init__()
        self.hidden_size = cfg.hidden_size #here hiddent size will be the size the two features join at (addition)
        
        # Unimodal reconstructors
        self.reconstructor_muys = nn.ModuleList([Reconstructor(self.hidden_size, self.hidden_size) for _ in range(cfg.num_modalities)])

        self.reconstructor_sigma2s = nn.ModuleList([ReconstructorSigma2(self.hidden_size, self.hidden_size) for _ in range(cfg.num_modalities)])
        
        # Multimodal mutli-head self attention
        self.attn = nn.MultiheadAttention(cfg.hidden_size, num_heads=2, dropout=0, batch_first=True)

        # feed forward network
        self.joint_processing = IMU_MLP(cfg.hidden_size, cfg.hidden_size//2, cfg.num_classes)
        self.joint_uncertainty = OutputSigma2(cfg.hidden_size, 1, cfg.num_classes)

    def forward(self, x, out_uncertainty=False):
        x, mask = x[:-1], x[-1]
        
        # reconstruction logic
        batch_muys = {k: {} for k in range(len(x))}
        batch_sigma2s = {k: {} for k in range(len(x))}
        batch_reps = []
        for i in range(len(x)):
            xs = []
            batch_rep_i = torch.zeros_like(x[i])
            batch_rep_i[mask[:, i]] = x[i][mask[:, i]] # if modality available, use the original data
            for j in range(len(x)):
                if i == j:
                    continue
                else:
                    rec_i_j = self.reconstructor_muys[i](x[j])
                    sigma2_i_j = self.reconstructor_sigma2s[i](x[j], rec_i_j) 
                   
                    batch_muys[i][j] = rec_i_j
                    batch_sigma2s[i][j] = sigma2_i_j

                    if out_uncertainty:
                        rec_i_j.requires_grad = True
                        rec_i_j.retain_grad()
                   
                    xs.append(rec_i_j * mask[:, j].unsqueeze(-1))
        
            divide_ratio = mask.sum(dim=1).float() - mask[:, i].float()
            batch_rep_i[~mask[:, i]] = (torch.stack(xs).sum(dim=0)[~mask[:, i]] / divide_ratio.unsqueeze(-1)[~mask[:, i]]) # if not available, use the average reconstructed data from other modalities

            batch_reps.append(batch_rep_i)
        
        z_features = torch.stack(batch_reps, dim=1) # 
        
        # attention and output
        out, _ = self.attn(z_features, z_features, z_features)
        # batch, seq, features = out.shape
        out = out.sum(dim=1)
        result = self.joint_processing(out)
        fused_uncertainty = self.joint_uncertainty(out, result)
        
        return result, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s