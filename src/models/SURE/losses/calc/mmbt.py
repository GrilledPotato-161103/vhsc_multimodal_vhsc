import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SURE.modules.mmbt import (
    UnimodalBertEncoder, Reconstructor, ReconstructUncertainty, OutputSigma2
)
from models.SURE.modules.gmc_module import AffectEncoder
from models.SURE.trainers.model_evaluation_metrics import *
from models.SURE.losses.nce_loss import NCELoss, GaussianAlignLoss, OrderedEnforceLoss, WeightedCrossEntropyLoss

def metrics_calculate(prediction, 
                        target, 
                        fused_uncertainty, 
                        target_mask, 
                        batch_representations, 
                        reconstruct_muys, 
                        reconstruct_sigma2s, 
                        beta,
                        use_fused_uncertainty=False,
                        criterion=nn.CrossEntropyLoss(),
                        reconstruct_criterion=GaussianAlignLoss(),
                        output_criterion=GaussianAlignLoss(task='classification'),
                        ):
        joint_mod_loss_sum = torch.tensor(0., device=prediction.device)
        target_mask = target_mask.squeeze()
        whole_mask = torch.logical_and(target_mask[:, 0], target_mask[:, 1])
        partial_mask = torch.logical_or(target_mask[:, 0], target_mask[:, 1])
        
        if not use_fused_uncertainty:
            if torch.any(whole_mask):
                for mod in range(len(reconstruct_muys)):
                    joint_mod_loss_sum += reconstruct_criterion(batch_representations[mod][whole_mask], reconstruct_muys[mod][whole_mask], reconstruct_sigma2s[mod][whole_mask], mod)

                joint_mod_loss_sum /= len(reconstruct_muys)
                loss = torch.mean(joint_mod_loss_sum)
            else:
                loss = joint_mod_loss_sum
        else:
            reconstruct_loss = criterion(prediction[partial_mask], target[partial_mask])
            try:
                reconstruct_loss.backward(retain_graph=True)
                # calculate the propagation uncertainty
                propagate_uncertainty = torch.zeros_like(fused_uncertainty)
                for mod in range(len(reconstruct_muys)):
                    mask_i = torch.logical_and(~target_mask[:, mod], partial_mask)
                    # print(reconstruct_muys[mod].grad[mask_i].norm())
                    propagate_uncertainty[mask_i] = ((reconstruct_muys[mod].grad[mask_i])**2 * reconstruct_sigma2s[mod][mask_i]).sum(dim=-1, keepdim=True)
                    reconstruct_muys[mod].grad = None # clear the gradient
                    torch.cuda.empty_cache()
                
                # print('propagate_uncertainty: ', propagate_uncertainty.mean(), 'fused_uncertainty: ', fused_uncertainty.mean())
                fused_uncertainty += propagate_uncertainty
            except:
                pass
            joint_mod_loss_sum = output_criterion(target, prediction, fused_uncertainty, -1, beta=beta)
            
            supervised_loss = criterion(prediction[partial_mask], target[partial_mask])

            loss = torch.mean(supervised_loss + joint_mod_loss_sum)

        
        return loss