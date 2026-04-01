import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SURE.modules.mmbt import (
    UnimodalBertEncoder, Reconstructor, ReconstructUncertainty, OutputSigma2
)
from models.SURE.modules.gmc_module import AffectEncoder
from models.SURE.trainers.model_evaluation_metrics import *
from models.SURE.losses.nce_loss import NCELoss, GaussianAlignLoss, OrderedEnforceLoss, WeightedCrossEntropyLoss

class SuperMMBT(nn.Module):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", dataset='mosi', use_fast_loading=False):
        super(SuperMMBT, self).__init__()
        # hyper-parameters
        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.dataset = dataset
        self.use_fast_loading = use_fast_loading

        # architecture
        self.image_processor = None
        self.text_processor = None
        self.processor = None
        self.reconstructor_shared = []
        self.reconstructor_muys = []
        self.reconstructor_sigma2s = []
        self.classifier = None
        
        # loss functions
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = WeightedCrossEntropyLoss()
        self.reconstruct_criterion = GaussianAlignLoss()
        self.output_criterion = GaussianAlignLoss(task='classification')
        self.order_criterion = OrderedEnforceLoss()


    def encode(self, x, return_reps=False):
        # If we have complete observations
        if not self.use_fast_loading:
            x_, mask = x[:-1], x[-1]
            img_rep, txt_rep, attn = self.processor(x_[0], x_[1], x_[2], x_[3])
        else:
            txt_rep, img_rep, attn, mask = x
        txt_muy, img_muy = self.reconstructor_muys[0](self.reconstructor_shared[0](img_rep)), self.reconstructor_muys[1](self.reconstructor_shared[1](txt_rep))
        # txt_muy = F.pad(txt_muy, (0,0,0, txt_len - img_len, 0, 0))
        # img_muy = img_muy[:, :img_len, :]

        img_rep = img_rep * mask[:, 0] + img_muy * ~(mask[:, 0])
        txt_rep = txt_rep * mask[:, 1] + txt_muy * ~(mask[:, 1])

        input_rep = torch.cat([img_rep, txt_rep], dim=1)
        joint_rep = self.pooler(self.fuser(input_rep, attn, output_all_encoded_layers=False)[-1])
        
        output = self.classifier(joint_rep)
        return output, self.fused_uncertainty(joint_rep, output)

    def forward(self, x, out_uncertainty=False, save_embedding=False):
        # Forward pass through the modality specific encoders
        if not self.use_fast_loading:
            x_, mask = x[:-1], x[-1]

            img_representation, txt_reprensentation, attn = self.processor(x_[0], x_[1], x_[2], x_[3])
            if save_embedding:
                return img_representation, txt_reprensentation, attn
        else:
            txt_reprensentation, img_representation, attn, mask = x
        # reconstruct logic
        txt_reconstruct_muys, img_reconstruct_muys = self.reconstructor_muys[0](self.reconstructor_shared[0](img_representation)), self.reconstructor_muys[1](self.reconstructor_shared[1](txt_reprensentation))
        if out_uncertainty:
            txt_reconstruct_muys.requires_grad = True
            img_reconstruct_muys.requires_grad = True
            txt_reconstruct_muys.retain_grad()
            img_reconstruct_muys.retain_grad()
        txt_reconstruct_sigma2, img_reconstruct_sigma2 = self.reconstructor_sigma2s[0](self.reconstructor_shared[0](img_representation), txt_reconstruct_muys), self.reconstructor_sigma2s[1](self.reconstructor_shared[1](txt_reprensentation), img_reconstruct_muys)
        
        # Forward pass through the joint encoder
        img_representation = img_representation * mask[:, 0] + img_reconstruct_muys * ~(mask[:, 0])
        txt_reprensentation = txt_reprensentation * mask[:, 1] + txt_reconstruct_muys * ~(mask[:, 1])

        input_representation = torch.cat([img_representation, txt_reprensentation], dim=1)
        joint_representation = self.pooler(self.fuser(input_representation, attn, output_all_encoded_layers=False)[-1])

        output = self.classifier(joint_representation)
        fused_uncertainty = self.fused_uncertainty(joint_representation, output)

        batch_reps = [img_representation, txt_reprensentation]
        batch_muys = [img_reconstruct_muys, txt_reconstruct_muys]
        batch_sigma2s = [img_reconstruct_sigma2, txt_reconstruct_sigma2]
        # import IPython; IPython.embed(); exit(1)
        return output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s

    def metrics_calculate(self, prediction, 
                                target, 
                                fused_uncertainty, 
                                target_mask, 
                                batch_representations, 
                                reconstruct_muys, 
                                reconstruct_sigma2s, 
                                temperature, 
                                beta, 
                                batch_size, 
                                epoch, 
                                use_fused_uncertainty=False):
        joint_mod_loss_sum = torch.tensor(0., device=prediction.device)
        target_mask = target_mask.squeeze()
        whole_mask = torch.logical_and(target_mask[:, 0], target_mask[:, 1])
        partial_mask = torch.logical_or(target_mask[:, 0], target_mask[:, 1])
        
        if not use_fused_uncertainty:
            if torch.any(whole_mask):
                for mod in range(len(reconstruct_muys)):
                    joint_mod_loss_sum += self.reconstruct_criterion(batch_representations[mod][whole_mask], reconstruct_muys[mod][whole_mask], reconstruct_sigma2s[mod][whole_mask], mod)

                joint_mod_loss_sum /= len(reconstruct_muys)
                loss = torch.mean(joint_mod_loss_sum)
            else:
                loss = joint_mod_loss_sum
        else:
            reconstruct_loss = self.criterion(prediction[partial_mask], target[partial_mask])
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
            joint_mod_loss_sum = self.output_criterion(target, prediction, fused_uncertainty, -1, beta=beta)
            
            supervised_loss = self.criterion(prediction[partial_mask], target[partial_mask])

            loss = torch.mean(supervised_loss + joint_mod_loss_sum)
        # loss = torch.mean(joint_mod_loss_sum)
        # loss = torch.mean(supervised_loss)

        if self.dataset == 'book':
            tqdm_dict = eval_book(prediction, target, fused_uncertainty)

        tqdm_dict["loss"] = loss
        
        return loss, tqdm_dict
    
    def save_embedding(self, data):
        img_representation, txt_reprensentation, attn = self.forward(data, save_embedding=True)
        return img_representation, txt_reprensentation, attn

    def training_step(self, data, target_data, mask_data, train_params, epoch, out_uncertainty=False, beta=0.5):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.forward(data, out_uncertainty)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.metrics_calculate(output, target_data, fused_uncertainty, mask_data, batch_reps, batch_muys, batch_sigma2s, temperature, beta, batch_size, epoch, out_uncertainty)

        return loss, tqdm_dict

    def validation_step(self, data, target_data, mask_data, train_params, epoch, out_uncertainty=False, beta=0.5):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.forward(data)
        # Compute contrastive loss
        loss, tqdm_dict = self.metrics_calculate(output, target_data, fused_uncertainty, mask_data, batch_reps, batch_muys, batch_sigma2s, temperature, beta, batch_size, epoch, out_uncertainty)

        # compute variance of the reconstruction over training process
        return tqdm_dict


class MMBT(SuperMMBT):
    def __init__(self, name, num_classes, common_dim, latent_dim, img_tokens, txt_tokens, dataset, checkpoint=None, args=None):
        super(MMBT, self).__init__(name, common_dim, latent_dim, use_fast_loading=args.use_fast_loading)
        self.name = name
        self.num_classes = num_classes
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.args = args
        
        self.processor = UnimodalBertEncoder(args)

        self.reconstructor_shared = nn.ModuleList([Reconstructor(img_tokens, img_tokens, common_dim=common_dim, latent_dim=common_dim), 
                                                 Reconstructor(txt_tokens, txt_tokens, common_dim=common_dim, latent_dim=common_dim)])
        
        # self.reconstructor_muys = nn.ModuleList([AffectEncoder(common_dim=common_dim, latent_dim=latent_dim) for _ in range(2)])
        self.reconstructor_muys = nn.ModuleList([Reconstructor(img_tokens, txt_tokens, common_dim=common_dim, latent_dim=latent_dim), 
                                                 Reconstructor(txt_tokens, img_tokens, common_dim=common_dim, latent_dim=latent_dim)])
        # self.reconstructor_sigma2s = nn.ModuleList([AffectEncoder(common_dim=common_dim, latent_dim=latent_dim, positive=True) for _ in range(2)])
        self.reconstructor_sigma2s = nn.ModuleList([ReconstructUncertainty(img_tokens, txt_tokens, common_dim=common_dim, latent_dim=latent_dim),
                                                    ReconstructUncertainty(txt_tokens, img_tokens, common_dim=common_dim, latent_dim=latent_dim)])
        
        self.fuser = self.processor.encoder
        self.pooler = self.processor.pooler
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.fused_uncertainty = OutputSigma2(in_dim=latent_dim, n_classes=num_classes, out_dim=1)
        
        self.__load_unimodal_projectors(checkpoint)

    def __load_unimodal_projectors(self, checkpoint):
        uni_ckpt = torch.load(checkpoint)
        try:
            # encoder_sd = {k.replace('enc.', ''): v for k, v in uni_ckpt['state_dict'].items() if 'enc.' in k}
            # self.processor.load_state_dict(encoder_sd, strict=False)
            self.load_state_dict(uni_ckpt, strict=False)
        except Exception as e:
            print('Error in loading the unimodal projectors: ', e)
            return
        print('Successfully load ', checkpoint)

        for param in self.processor.parameters():
            param.requires_grad = False