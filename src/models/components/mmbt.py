import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SURE.modules.mmbt import (
    UnimodalBertEncoder, Reconstructor, ReconstructUncertainty, OutputSigma2
)
from models.SURE.modules.gmc_module import AffectEncoder
from models.SURE.trainers.model_evaluation_metrics import *
from models.SURE.losses.nce_loss import NCELoss, GaussianAlignLoss, OrderedEnforceLoss, WeightedCrossEntropyLoss


class MMBT(nn.Module):
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
