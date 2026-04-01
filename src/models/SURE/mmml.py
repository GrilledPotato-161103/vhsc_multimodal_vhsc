import torch
import torch.nn as nn

from models.modules.context_model import rob_d2v_cc_context_reconstruct
from util_scripts.metricsTop import MetricsTop
from models.losses.nce_loss import NCELoss, GaussianAlignLoss, WeightedL1Loss, WeightedCrossEntropyLoss

class SuperMMML(nn.Module):
    def __init__(self, name, dataset='mosi', cfg=None):
        super(SuperMMML, self).__init__()
        # hyper-parameters
        self.name = name
        self.cfg = cfg
        self.model = None
        self.dataset = dataset
        # loss functions
        self.criterion = nn.L1Loss() if cfg.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.w_criterion = WeightedL1Loss() if cfg.train_mode == 'regression' else WeightedCrossEntropyLoss()
        self.maxlikelihood_criterion = GaussianAlignLoss(gamma=cfg.gamma)
        self.metrics = MetricsTop(cfg.train_mode).getMetics(cfg.dataset_name)
        self.tasks = cfg.tasks

    def encode(self, x, return_reps=False):
        # Forward pass through the encoders
        # text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, sample_mask = x
        
        # output, batch_reps, batch_muys, batch_sigma2s = self.forward(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, sample_mask)
        
        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.model(*x)
        return output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s

    def forward(self, x, out_uncertainty=False):
        return self.model(*x, out_uncertainty)

    def metrics_calculate(self, prediction, target, fused_uncertainty, target_mask, batch_representations, reconstruct_muys, reconstruct_sigma2s, temperature, beta, batch_size, epoch, use_fused_uncertainty=False):
        joint_mod_loss_sum = 0
        target_mask = target_mask.squeeze()
        whole_mask = torch.logical_and(target_mask[:, 0], target_mask[:, 1])
        partial_mask = torch.logical_or(target_mask[:, 0], target_mask[:, 1])
        
        if not use_fused_uncertainty:
            for mod in range(len(reconstruct_muys)):
                if torch.any(whole_mask):
                    mod_mle = self.maxlikelihood_criterion(batch_representations[mod][whole_mask], reconstruct_muys[mod][whole_mask], reconstruct_sigma2s[mod][whole_mask], mod, beta=1.)
                    # print('reconstruct loss for', mod, ': ', mod_mle.mean())
                    joint_mod_loss_sum += mod_mle

            loss = torch.mean(joint_mod_loss_sum)
        else:
            reconstruct_loss = ((prediction['M'] - target)**2).sum()
            try:
                reconstruct_loss.backward(retain_graph=True)
                # calculate the propagation uncertainty
                propagate_uncertainty = torch.zeros_like(fused_uncertainty)
                for mod in range(len(reconstruct_muys)):
                    mask_i = torch.logical_and(~target_mask[:, mod], partial_mask)
                    # print(reconstruct_muys[mod].grad[mask_i].norm())
                    propagate_uncertainty[mask_i] = ((reconstruct_muys[mod].grad[mask_i])**2 * reconstruct_sigma2s[mod][mask_i]).sum(dim=-1, keepdim=True)
                
                # print('propagate_uncertainty: ', propagate_uncertainty.mean(), 'fused_uncertainty: ', fused_uncertainty.mean())
                fused_uncertainty += propagate_uncertainty
            except:
                pass
            joint_mod_loss_sum = self.maxlikelihood_criterion(target, prediction['M'], fused_uncertainty, -1, beta=beta)
            supervised_loss = 0.0
            if self.cfg.multi_task:
                for m in self.tasks:
                    if m == 'M':
                        sub_loss = self.cfg.loss_weights[m] * self.w_criterion(prediction[m], target, target_mask, reconstruct_sigma2s)
                    else:
                        sub_loss = self.cfg.loss_weights[m] * self.criterion(prediction[m][partial_mask], target[partial_mask])
                    supervised_loss += sub_loss
            else:
                supervised_loss = self.criterion(prediction['M'][partial_mask], target[partial_mask])

            loss = torch.mean(supervised_loss + joint_mod_loss_sum)
        # loss = torch.mean(joint_mod_loss_sum)
        # loss = torch.mean(supervised_loss)

        if self.dataset == 'mmml_mosi':
            tqdm_dict = self.metrics(prediction['M'][partial_mask], target[partial_mask], fused_uncertainty[partial_mask])

        tqdm_dict["loss"] = loss
        return loss, tqdm_dict

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


class MMML(SuperMMML):
    def __init__(self, name, dataset='mosi', cfg=None):
        super(MMML, self).__init__(name, dataset, cfg)
        self.name = name

        self.model = rob_d2v_cc_context_reconstruct(cfg)
        
        self.__load_unimodal_projectors(cfg.frozen_checkpoint_path)

    def __load_unimodal_projectors(self, checkpoint):
        uni_ckpt = torch.load(checkpoint)
        encoder_sd = uni_ckpt
        try:
            # self.model.load_state_dict(encoder_sd, strict=False)
            self.load_state_dict(encoder_sd, strict=False)
            pass
        except Exception as e:
            print('Error in loading the unimodal projectors: ', checkpoint, e)
        print('Successfully load ', checkpoint)