import torch
import torch.nn as nn

from models.modules.hamlet_module import EmbeddingHAMLET as FusedModel
from models.trainers.model_evaluation_metrics import calculate_accuracy
from models.losses.nce_loss import NCELoss, GaussianAlignLoss, WeightedL1Loss, WeightedCrossEntropyLoss

class SuperHAMLET(nn.Module):
    def __init__(self, name, dataset='mosi', cfg=None):
        super(SuperHAMLET, self).__init__()
        # hyper-parameters
        self.name = name
        self.cfg = cfg
        self.model = None
        self.dataset = dataset
        # loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.reconstruct_criterion = GaussianAlignLoss(gamma=cfg.gamma)
        self.output_criterion = GaussianAlignLoss(gamma=cfg.gamma, task='classification')
        self.metrics = calculate_accuracy

    def encode(self, x, return_reps=False):
        # Forward pass through the encoders
        # text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, sample_mask = x
        
        # output, batch_reps, batch_muys, batch_sigma2s = self.forward(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, sample_mask)
        
        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.model(x)
        return output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s

    def forward(self, x, out_uncertainty=False, save_embedding=False):
        return self.model(x, out_uncertainty)

    def metrics_calculate(self, prediction, target, fused_uncertainty, target_mask, batch_representations, reconstruct_muys, reconstruct_sigma2s, temperature, beta, batch_size, epoch, use_fused_uncertainty=False):
        joint_mod_loss_sum = 0
        target_mask = target_mask.squeeze()
        if not use_fused_uncertainty:
            for i in reconstruct_muys.keys():
                for j in reconstruct_muys[i].keys():
                    mask_ij = target_mask[:, i] & target_mask[:, j]
                    if torch.any(mask_ij):
                        mod_mle = self.reconstruct_criterion(batch_representations[i][mask_ij], reconstruct_muys[i][j][mask_ij], reconstruct_sigma2s[i][j][mask_ij], i, beta=1)
                        joint_mod_loss_sum += mod_mle

            loss = torch.mean(joint_mod_loss_sum)
        else:
            reconstruct_loss = self.criterion(prediction, target)
            try:
                reconstruct_loss.backward(retain_graph=True)
                # calculate the propagation uncertainty
                propagate_uncertainty = torch.zeros_like(fused_uncertainty)
                for mod in reconstruct_muys.keys():
                    mask_i = ~target_mask[:, mod]
                    reconstruct_uncert_i = torch.stack([reconstruct_sigma2s[mod][j] for j in reconstruct_muys[mod].keys()]).sum(dim=0)
                    propagate_uncertainty[mask_i] = ((reconstruct_muys[mod].grad[mask_i])**2 * reconstruct_uncert_i[mask_i]).sum(dim=-1, keepdim=True)
                
                print('propagate_uncertainty: ', propagate_uncertainty.mean(), 'fused_uncertainty: ', fused_uncertainty.mean())
                fused_uncertainty += propagate_uncertainty
            except:
                pass
            joint_mod_loss_sum = self.output_criterion(target, prediction, fused_uncertainty, -1, beta=self.cfg.beta)
            supervised_loss = 0.0
            supervised_loss = self.criterion(prediction, target)

            loss = torch.mean(supervised_loss + joint_mod_loss_sum)

        if self.dataset == 'mmact':
            tqdm_dict = self.metrics(prediction, target, fused_uncertainty)

        tqdm_dict["loss"] = loss
        return loss, tqdm_dict

    def save_embedding(self, data):
        rgb_ebd, watch_accel_ebd, phone_gyro_ebd = self.forward(data, save_embedding=True)
        return rgb_ebd, watch_accel_ebd, phone_gyro_ebd
    
    def training_step(self, data, target_data, mask_data, train_params, epoch, out_uncertainty=False, beta=1.):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.forward(data, out_uncertainty)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.metrics_calculate(output, target_data, fused_uncertainty, mask_data, batch_reps, batch_muys, batch_sigma2s, temperature, beta, batch_size, epoch, out_uncertainty)

        return loss, tqdm_dict

    def validation_step(self, data, target_data, mask_data, train_params, epoch, out_uncertainty=False, beta=1.):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.forward(data, out_uncertainty)
        # Compute contrastive loss
        loss, tqdm_dict = self.metrics_calculate(output, target_data, fused_uncertainty, mask_data, batch_reps, batch_muys, batch_sigma2s, temperature, beta, batch_size, epoch, out_uncertainty)

        # compute variance of the reconstruction over training process
        return tqdm_dict


class HAMLET(SuperHAMLET):
    def __init__(self, name, dataset='mosi', cfg=None):
        super(HAMLET, self).__init__(name, dataset, cfg)
        self.name = name

        self.model = FusedModel(cfg)
        self.__load_unimodal_projectors(cfg.frozen_checkpoint_path)

    def __load_unimodal_projectors(self, checkpoint):
        uni_ckpt = torch.load(checkpoint)
        try:
            # self.model.load_state_dict(encoder_sd, strict=False)
            state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in uni_ckpt.items() if k in state_dict and v.size() == state_dict[k].size()}
            self.load_state_dict(filtered_state_dict, strict=False)
            # print('After loading the unimodal projectors')
            # print(self.model.joint_processing.layers[0].layers[0].weight)
            pass
        except Exception as e:
            print('Error in loading the unimodal projectors: ', checkpoint, e)
        print('Successfully load ', checkpoint)