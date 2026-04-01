import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.gmc_module import (AffectGRUEncoder, 
                                       AffectJointProcessor, 
                                        AffectJointProcessor2,
                                       AffectEncoder)
from models.trainers.model_evaluation_metrics import *
from models.losses.nce_loss import NCELoss, GaussianAlignLoss

class SuperGMC(nn.Module):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", dataset='mosi'):
        super(SuperGMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.dataset = dataset

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None
        self.proj1 = None
        self.proj2 = None
        self.classifier = None
        self.criterion = nn.L1Loss()
        self.maxlikelihood_criterion = GaussianAlignLoss()


    def encode(self, x, return_reps=False):

        # If we have complete observations
        if None not in x:
            batch_projections = []
            for processor_idx in range(len(self.processors) - 1):
                mod_projection = self.processors[processor_idx](x[processor_idx])
                batch_projections.append(mod_projection)

            joint_representation = self.encoder(self.processors[-1](batch_projections))
            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
            output += joint_representation
            if return_reps:
                return output
            return self.classifier(output)
        else:
            if x[0] is None:
                latent1 = self.processors[1](x[1])
                latent0 = self.reconstructor_muys[0](latent1)
            elif x[1] is None:
                latent0 = self.processors[0](x[0])
                latent1 = self.reconstructor_muys[1](latent0)

            latent = self.encoder(self.processors[-1]([latent0, latent1]))
            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(latent)), p=0.0, training=self.training))
            output += latent
            if return_reps:
                return output
            return self.classifier(output)

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_projections = []
        outputs = []
        for processor_idx in range(len(self.processors) - 1):
            mod_projection = self.processors[processor_idx](x[processor_idx])
            batch_projections.append(mod_projection)

        reconstruct_muys = [self.reconstructor_muys[0](batch_projections[1]), self.reconstructor_muys[1](batch_projections[0])]
        reconstruct_sigma2s = [self.reconstructor_sigma2s[0](batch_projections[1]), self.reconstructor_sigma2s[1](batch_projections[0])]

        # Forward pass through the joint encoder
        joint_representation_0 = self.encoder(self.processors[-1]([reconstruct_muys[0], batch_projections[1]]))
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation_0)), p=0.0, training=self.training))
        output += joint_representation_0
        outputs.append(self.classifier(output))

        joint_representation_1 = self.encoder(self.processors[-1]([batch_projections[0], reconstruct_muys[1]]))
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation_1)), p=0.0, training=self.training))
        output += joint_representation_1
        outputs.append(self.classifier(output))

        return outputs, batch_projections, reconstruct_muys, reconstruct_sigma2s

    def super_gmc_loss(self, prediction, target, target_mask, batch_representations, reconstruct_muys, reconstruct_sigma2s, temperature, batch_size, epoch):
        joint_mod_loss_sum = 0
        supervised_loss = 0
        for mod in range(len(reconstruct_muys)):
            loss_joint_mod = self.maxlikelihood_criterion(batch_representations[mod], reconstruct_muys[mod], reconstruct_sigma2s[mod])
            joint_mod_loss_sum += loss_joint_mod
            if torch.any(target_mask):
                supervised_loss += self.criterion(prediction[mod][target_mask], target[target_mask])
            else:
                supervised_loss += torch.tensor(0.0, device=target.device)
        
        joint_mod_loss_sum /= len(reconstruct_muys)
        supervised_loss /= len(reconstruct_muys)
        
        # import IPython; IPython.embed(); exit(1)
        loss = torch.mean(joint_mod_loss_sum + supervised_loss)

        if self.dataset == 'mosei':
            tqdm_dict = eval_mosei(prediction, target, exclude_zero=True)
        else:
            tqdm_dict = eval_mosi(prediction[0], target, exclude_zero=True)
        # loss = torch.mean(supervised_loss)
        tqdm_dict["loss"] = loss
        
        return loss, tqdm_dict

    def training_step(self, data, target_data, mask_data, train_params, epoch):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations, reconstruct_muys, reconstruct_sigma2s = self.forward(data)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, mask_data, batch_representations, reconstruct_muys, reconstruct_sigma2s, temperature, batch_size, epoch)

        return loss, tqdm_dict

    def validation_step(self, data, target_data, mask_data, train_params, epoch):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations, reconstruct_muys, reconstruct_sigma2s = self.forward(data)
        # Compute contrastive loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, mask_data, batch_representations, reconstruct_muys, reconstruct_sigma2s, temperature, batch_size, epoch)

        return tqdm_dict

# Affect
class AffectGMC(SuperGMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", dataset='mosei', transfer=False, checkpoint=None):
        super(AffectGMC, self).__init__(name, common_dim, latent_dim, loss_type)

        if not transfer:
            if dataset == 'mosei':
                self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
                self.audio_processor = AffectGRUEncoder(input_dim=74, hidden_dim=30, latent_dim=latent_dim, timestep=50)
                # self.vision_processor = AffectGRUEncoder(input_dim=35, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            else:
                self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
                self.audio_processor = AffectGRUEncoder(input_dim=5, hidden_dim=30, latent_dim=latent_dim, timestep=50)
                # self.vision_processor = AffectGRUEncoder(input_dim=20, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.joint_processor = AffectJointProcessor2(latent_dim, dataset)
        else:
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=5, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=20, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.joint_processor = AffectJointProcessor(latent_dim, 'mosi')
        
        self.transfer = transfer
        self.processors = [
            self.language_processor,
            self.audio_processor,
            # self.vision_processor,
            self.joint_processor]

        self.loss_type = loss_type

        self.reconstructor_muys = nn.ModuleList([AffectEncoder(common_dim=latent_dim, latent_dim=latent_dim) 
                                          for _ in range(2)])
        self.reconstructor_sigma2s = nn.ModuleList([AffectEncoder(common_dim=latent_dim, latent_dim=latent_dim, positive=True) 
                                          for _ in range(2)])

        self.encoder = AffectEncoder(common_dim=common_dim, latent_dim=latent_dim)

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)
        # print(self.joint_processor)
        if checkpoint is not None:
            self.__load_unimodal_projectors(checkpoint)

    def __load_unimodal_projectors(self, checkpoint):
        uni_ckpt = torch.load(checkpoint)
        self.load_state_dict(uni_ckpt, strict=False)

        for param in self.parameters():
            param.requires_grad = False

        # only train the projectors
        for param in [self.reconstructor_muys.parameters(), self.reconstructor_sigma2s.parameters()]:
            for p in param:
                p.requires_grad = True
        