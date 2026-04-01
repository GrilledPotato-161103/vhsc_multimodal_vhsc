import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.gmc_module import AffectEncoder
from models.modules.vilt_module import (TextProcessor, ImageProcessor, JointProcessor)
from models.trainers.model_evaluation_metrics import *
from models.losses.nce_loss import NCELoss

class SuperViLT(nn.Module):
    def __init__(self, name, common_dim, latent_dim, dataset='mosi'):
        super(SuperViLT, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
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
        self.nce_critertion = NCELoss(dataset=dataset)
        if self.dataset in ['mmimdb']:
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.dataset in ['food101', 'hatememes']:
            self.criterion = nn.CrossEntropyLoss()


    def encode(self, x, sample=False, return_reps=False):

        # If we have complete observations
        if None not in x:

            joint_representation = self.encoder(self.processors[-1](x))
            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
            output += joint_representation
            if return_reps:
                return output
            return self.classifier(output)

        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)

            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(latent)), p=0.0, training=self.training))
            output += latent
            if return_reps:
                return output
            return self.classifier(output)

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
        output += joint_representation

        return self.classifier(output), batch_representations

    def super_gmc_loss(self, prediction, target, target_mask, batch_representations, temperature, batch_size, epoch):
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1) / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod
        
        if torch.any(target_mask):
            if self.dataset == 'mmimdb':
                supervised_loss = self.criterion(prediction[target_mask], target[target_mask])
            elif self.dataset in ['food101', 'hatememes']:
                supervised_loss = self.criterion(prediction[target_mask], target[target_mask].long())
        else:
            supervised_loss = torch.tensor(0.0, device=target.device)

        loss = torch.mean(joint_mod_loss_sum + supervised_loss)

        if self.dataset == 'mmimdb':
            tqdm_dict = calculate_f1(prediction, target)
        elif self.dataset == 'food101':
            tqdm_dict = calculate_accuracy(prediction, target)
        elif self.dataset == 'hatememes':
            tqdm_dict = calculate_auroc(prediction, target)
        # loss = torch.mean(supervised_loss)
        tqdm_dict["loss"] = loss
        
        return loss, tqdm_dict

    def training_step(self, data, target_data, mask_data, train_params, epoch):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, mask_data, batch_representations, temperature, batch_size, epoch)

        return loss, tqdm_dict

    def validation_step(self, data, target_data, mask_data, train_params, epoch):
        temperature = train_params.temperature
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)
        # Compute contrastive loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, mask_data, batch_representations, temperature, batch_size, epoch)
        return tqdm_dict




# Affect
class ViLT(SuperViLT):
    def __init__(self, name, num_classes, common_dim, latent_dim, dataset='ucf101'):
        super(ViLT, self).__init__(name, common_dim, latent_dim, dataset=dataset)

        if dataset in ['mmimdb', 'food101', 'hatememes']:
            self.text_processor = TextProcessor(common_dim=common_dim, latent_dim=latent_dim)
            self.image_processor = ImageProcessor(common_dim=common_dim, latent_dim=latent_dim)

            self.joint_processor = JointProcessor(common_dim=common_dim, latent_dim=latent_dim)

        self.processors = nn.ModuleList([
            self.text_processor,
            self.image_processor,
            self.joint_processor])

        self.encoder = AffectEncoder(common_dim=latent_dim, latent_dim=latent_dim)

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
    