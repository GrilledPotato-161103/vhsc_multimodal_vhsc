import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel, HubertModel, AutoModel, Data2VecAudioModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English text model + context
class roberta_en_context(nn.Module):
    def __init__(self):        
        super().__init__() 
        self.roberta_model = AutoModel.from_pretrained('roberta-large') # with context, we can improve using a larger model
        self.classifier = nn.Linear(1024*2, 1)    
   
    def forward(self, input_ids, attention_mask, context_input_ids, context_attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        context_output = self.roberta_model(context_input_ids, context_attention_mask, return_dict=True)
        context_pooler = context_output["pooler_output"]   # Shape is [batch_size, 1024]

        pooler = torch.cat((input_pooler, context_pooler), dim=1)
        output = self.classifier(pooler)                    # Shape is [batch_size, 1]
        return output
    

# English text+audio model + context
class rob_d2v_cc_context(nn.Module):
    def __init__(self, config):        
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1)
           )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
          )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2+768*2, 1024*2),
            nn.ReLU(),
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
        
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        context_pooler = raw_output_context["pooler_output"]    # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0]!=0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features,0).to(device)   # Shape is [batch_size, 768]
        
        # audio context feature extraction
        audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        A_context_hidden_states = audio_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_context_out.attentions[layer][batch][0][0]!=0)
                    audio_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features,0).to(device)   # Shape is [batch_size, 768]

        T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 768*2]

        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_features, A_features), dim=1)    # Shape is [batch_size, 1024*2+768*2]
        fused_output = self.fused_output_layers(fused_features)        # Shape is [batch_size, 1]

        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }

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
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim)
        )

    def forward(self, x, rec):
        x = torch.cat((x, rec), dim=-1)
        x = self.net(x)
        x = F.softplus(x)
        # x = F.normalize(x, p=2, dim=-1)
        # x = x / torch.norm(x, p=2, dim=-1, keepdim=True) # normalize the output to have unit norm
        return x
    
class OutputSigma2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(OutputSigma2, self).__init__()
        self.net_1 = nn.Sequential(
            nn.Linear(in_dim + 1, in_dim // 2),
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
    
# English text+audio model + context
class rob_d2v_cc_context_reconstruct(nn.Module):            
    def __init__(self, config=None):        
        super().__init__()
        # # encoders
        dropout = config.dropout if config is not None else 0.1
        # reconstructors
        self.reconstruct_muy_T = Reconstructor(768*2, 1024*2)
        self.reconstruct_sigma2_T = ReconstructorSigma2(768*2 + 1024*2, 1024*2)
        self.reconstruct_muy_A = Reconstructor(1024*2, 768*2)
        self.reconstruct_sigma2_A = ReconstructorSigma2(1024*2 + 768*2, 768*2)
        
        # output layers
        self.T_output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024*2, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768*2, 1)
          )
        self.preoutput_fused_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024*2+768*2, 1024*2),
            nn.ReLU(),
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
        )
        self.fused_output_layer = nn.Linear(1024, 1)
        
        self.fused_uncertainty = OutputSigma2(1024, 1)
        # self.uncertainty = nn.Parameter(torch.zeros(1024, 1), requires_grad=True)
        
    def forward(self, T_features, A_features, sample_mask, idxs, output_uncertainty=False):
        # if model is missing, reconstruct from here
        T_reconstruct = self.reconstruct_muy_T(A_features)    # Shape is [batch_size, 1024*2]
        T_rec_sigma2 = self.reconstruct_sigma2_T(A_features, T_reconstruct)    # Shape is [batch_size, 1024*2]
        A_reconstruct = self.reconstruct_muy_A(T_features)    # Shape is [batch_size, 768*2]
        A_rec_sigma2 = self.reconstruct_sigma2_A(T_features, A_reconstruct)    # Shape is [batch_size, 768*2]

        if output_uncertainty:
            T_reconstruct.requires_grad = True
            A_reconstruct.requires_grad = True
            T_reconstruct.retain_grad()
            A_reconstruct.retain_grad()

        T_final_ebd = T_features * sample_mask[:, 0] + T_reconstruct * ~(sample_mask[:, 0])
        A_final_ebd = A_features * sample_mask[:, 1] + A_reconstruct * ~(sample_mask[:, 1])
        T_output = self.T_output_layers(T_final_ebd)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_final_ebd)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_final_ebd, A_final_ebd), dim=1)    # Shape is [batch_size, 1024*2+768*2]
        
        # fused_sigma2 = T_rec_sigma2 * ~(sample_mask[:, 0]) + A_rec_sigma2 * ~(sample_mask[:, 1])
        # identity = torch.fill(torch.zeros_like(fused_sigma2), -1)
        # identity[~(sample_mask[:, 0])] = 1
        # identity[~(sample_mask[:, 1])] = 0
        
        fused_features_s = self.preoutput_fused_layers(fused_features)
        fused_output = self.fused_output_layer(fused_features_s)        # Shape is [batch_size, 1]
        fused_uncert = self.fused_uncertainty(fused_features_s, fused_output)

        # fused_uncertainty = torch.log(1 + torch.exp(self.uncertainty[idxs]))
        output = {
                'T': T_output,
                'A': A_output,
                'M': fused_output
        }
        batch_reps = [T_final_ebd, A_final_ebd]
        batch_muys = [T_reconstruct, A_reconstruct]
        batch_sigma2s = [T_rec_sigma2, A_rec_sigma2]
        return output, fused_uncert, batch_reps, batch_muys, batch_sigma2s