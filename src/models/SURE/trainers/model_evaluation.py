from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
import numpy as np
from .tasks._eval_protocols import *
from collections import defaultdict
from itertools import combinations
import pickle as pkl
import matplotlib.pyplot as plt
import json
import os
import copy

from models.trainers.model_evaluation_metrics import *
from util_scripts.wandb_logger import WandbLogger
from util_scripts.train_callbacks import ModelSaverLoaderCallback
from models.losses.nce_loss import GaussianAlignLoss

def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs, k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results

class ModelEvaluation(nn.Module):
    def __init__(self, model, dataset, test_loader, opt, modalities=None, class_names=None, train_loader=None, last_cpkt=False):
        super(ModelEvaluation, self).__init__()

        self.dataset = dataset

        self.test_modalities = modalities
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.class_names = class_names
        self.opt = opt
        # device
        self.device = opt.device
        # logger
        self.logger = WandbLogger(opt)
        # callback
        self.callback = ModelSaverLoaderCallback(opt.result_path, 'model', opt=opt)
        self.model = self.callback.load_cpkt(model, last=last_cpkt)
        # self.model = model
        self.theta_ast = Params2Vec(self.model.parameters()).to('cpu')
        self.labeled_ratio = opt.labeled_ratio
        self.model.eval()

        # missing modalities combinations
        modes = []
        modalities = list(range(self.test_modalities))
        for i in range(1, len(modalities) + 1):
            modes.extend(list(combinations(modalities, i)))
        
        self.modes = modes
        self.maxlikelihood_criterion = GaussianAlignLoss(opt.dataset, gamma=opt.gamma)
        
    def evaluate(self):
        with torch.no_grad():
            for mods in self.modes:
                # if mods != (0, 1):
                #     continue
                print('Evaluating modalities: ', mods) 
                if self.dataset in ['mosei', 'mosi']:
                    if self.labeled_ratio == 0.0:
                        train_reps = []
                        train_truths = []

                        for i_batch, (batch_X, batch_Y, batch_META) in tqdm(enumerate(self.train_loader)):
                            sample_ind, text, audio, vision = batch_X
                            data = [text.to(self.device), audio.to(self.device), vision.to(self.device)]
                            target_data, _ = batch_Y
                            target_data = target_data.squeeze(-1).to(self.device)
                            
                            # Drop modalities (if required)
                            input_data = []
                            
                            for j in range(len(data)):
                                if j not in mods:
                                    input_data.append(None)
                                else:
                                    input_data.append(data[j])

                            rep = self.model.encode(input_data, return_reps=True)
                            train_reps.append(rep.detach().cpu().numpy())
                            train_truths.append(target_data.detach().cpu().numpy())
                        
                        # train classifier
                        train_reps = np.concatenate(train_reps, axis=0)
                        train_truths = np.concatenate(train_truths, axis=0)
                        classifier = fit_lr(train_reps, train_truths.squeeze(-1))             
                                 
                    results = []
                    truths = []
                    for i_batch, (batch_X, batch_Y, batch_META) in tqdm(enumerate(self.test_loader)):
                        sample_ind, text, audio, vision = batch_X
                        # data = [text.to(self.device), audio.to(self.device), vision.to(self.device)]
                        data = [text.to(self.device), audio.to(self.device)]
                        target_data, _ = batch_Y
                        target_data = target_data.squeeze(-1).to(self.device)  # if num of labels is 1

                        # Drop modalities (if required)
                        input_data = []
                        
                        for j in range(len(data)):
                            if j not in mods:
                                input_data.append(None)
                            else:
                                input_data.append(data[j])

                        # Parallel model
                        if self.labeled_ratio != 0.0:
                            preds = self.model.encode(input_data)
                        else:
                            rep = self.model.encode(input_data, return_reps=True)
                            preds = torch.from_numpy(classifier.predict(rep.cpu().detach().numpy()))

                        # Collect the results into dictionary
                        truths.append(target_data)
                        results.append(preds)

                    results = torch.cat(results)
                    truths = torch.cat(truths)

                    if self.dataset == "mosei":
                        eval_mosei(results, truths, self.logger, True)
                    elif self.dataset == 'mosi':
                        eval_mosi(results, truths, self.logger, True)
                elif self.dataset in ['book']:
                    results = []
                    truths = []
                    sigma2s_out = []

                    reps, muys, sigma2s  = [], [], []
                    for i_batch, batch in tqdm(enumerate(self.test_loader)):
                        if self.opt.use_fast_loading:
                            txt, img, att, target, mask = batch
                            target_data = target.to(self.device)
                            
                            # Drop modalities (if required)
                            input_data = []
                            if mods == (0,):
                                mask[:, 1] = 0
                            elif mods == (1,):
                                mask[:, 0] = 0
                            
                            input_data = [txt.to(self.device), img.to(self.device), att.to(self.device), mask.unsqueeze(-1).unsqueeze(-1).to(self.device)]
                        else:
                            text, segment, mask, image, target_data, mask_data = batch
                            target_data = target_data.to(self.device)  # if num of labels is 1

                            # Drop modalities (if required)
                            input_data = []
                            if mods == (0,):
                                mask_data[:, 1] = 0
                            elif mods == (1,):
                                mask_data[:, 0] = 0
                            
                            input_data = [text.to(self.device), mask.to(self.device), segment.to(self.device), image.to(self.device), mask_data.unsqueeze(-1).unsqueeze(-1).to(self.device)]

                        output, pred_uncerts, batch_reps, batch_muys, batch_sigma2s = self.model.forward(input_data)
                        reps.append(batch_reps)
                        muys.append(batch_muys)
                        sigma2s.append(batch_sigma2s)

                        truths.append(target_data)
                        results.append(output)
                        sigma2s_out.append(pred_uncerts)

                    results = torch.cat(results)
                    truths = torch.cat(truths)
                    sigma2s_out = torch.cat(sigma2s_out)

                    reps = [torch.cat([r[0] for r in reps]), torch.cat([r[1] for r in reps])]
                    muys = [torch.cat([r[0] for r in muys]), torch.cat([r[1] for r in muys])]
                    sigma2s = [torch.cat([r[0] for r in sigma2s]), torch.cat([r[1] for r in sigma2s])]
                    self._vis_sigma(reps, muys, sigma2s)

                    if self.dataset == "book":
                        eval_book(results, truths, sigma2s_out, self.logger)
                elif self.dataset in ['mmact', 'utd_mhad']:
                    results = []
                    truths = []
                    
                    reps, muys, sigma2s = [], [], []
                    sigma2_outputs = []
                    for i_batch, batch in tqdm(enumerate(self.test_loader)):
                        video_inputs, accel_inputs, gyro_inputs, target_data, mask_data = batch
                        target_data = target_data.to(self.device)  # if num of labels is 1

                        # Drop modalities (if required)
                        input_data = []
                        mask_data = torch.zeros_like(mask_data, dtype=torch.bool)
                        for k in mods:
                            mask_data[:, k] = True
                        
                        input_data = [video_inputs.to(self.device), accel_inputs.to(self.device), gyro_inputs.to(self.device), mask_data.to(self.device)]


                        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.model.forward(input_data)
                        reps.append(batch_reps)
                        muys.append(batch_muys)
                        sigma2s.append(batch_sigma2s)

                        sigma2_outputs.append(fused_uncertainty)
                        # Collect the results into dictionary
                        truths.append(target_data)
                        results.append(output)


                    results = torch.cat(results)
                    truths = torch.cat(truths)
                    calculate_accuracy(results, truths, logger=self.logger)

                    self._vis_sigma_multimodal(reps, muys, sigma2s)
                    
                    sigma2_outputs = torch.cat(sigma2_outputs).squeeze()
                    err2 = F.cross_entropy(results, truths, reduction='none')
                    cov = (err2 - err2.mean(axis=0, keepdims=True)) * (sigma2_outputs - sigma2_outputs.mean(axis=0, keepdims=True))
                    pearson_corr = cov.mean(axis=0) / (err2.std(axis=0) * sigma2_outputs.std(axis=0) + 1e-5)
                    print('Output uncertainty corr w. error:', pearson_corr.mean())
                    print('Mutual Information err2 with sigma2: ', compute_mi(err2.detach().cpu().numpy(), sigma2_outputs.detach().cpu().numpy()), '/', compute_mi(err2.detach().cpu().numpy(), err2.detach().cpu().numpy()))
                elif self.dataset in ['mmml_mosi']:  
                    results = []
                    truths = []

                    reps, muys, sigma2s = [], [], []
                    sigma2_outputs = []
                    for i_batch, batch in tqdm(enumerate(self.test_loader)):
                        text_inputs = batch["reps_T"].to(self.device)
                        audio_inputs = batch["reps_A"].to(self.device)
                        idx_data = batch["index"].to(self.device)

                        target_data = batch["label"].to(self.device)
                        mask_data = batch["sample_mask"].unsqueeze(-1).to(self.device)
                        
                        if mods == (0,):
                            mask_data[:, 1] = False
                        elif mods == (1,):
                            mask_data[:, 0] = False
                        data = [text_inputs, audio_inputs, mask_data, idx_data]
                        
                        # Parallel model
                        output, fused_uncertainty, batch_reps, batch_muys, batch_sigma2s = self.model.forward(data)
                        reps.append(batch_reps)
                        muys.append(batch_muys)
                        sigma2s.append(batch_sigma2s)

                        # Collect the results into dictionary
                        sigma2_outputs.append(fused_uncertainty)
                        truths.append(target_data)
                        results.append(output['M'])
                        

                    results = torch.cat(results)
                    truths = torch.cat(truths)
                    
                    if self.dataset == "mmml_mosi":
                        print(eval_mosei(results, truths, self.logger, True))

                    reps = [torch.cat([r[0] for r in reps]), torch.cat([r[1] for r in reps])]
                    muys = [torch.cat([r[0] for r in muys]), torch.cat([r[1] for r in muys])]
                    sigma2s = [torch.cat([r[0] for r in sigma2s]), torch.cat([r[1] for r in sigma2s])]
                    self._vis_sigma_single_dim(reps, muys, sigma2s)
                    
                    sigma2_outputs = torch.cat(sigma2_outputs).squeeze()
                    err2 = ((truths - results)**2).mean(axis=-1)
                    cov = (err2 - err2.mean(axis=0, keepdims=True)) * (sigma2_outputs - sigma2_outputs.mean(axis=0, keepdims=True))
                    pearson_corr = cov.mean(axis=0) / (err2.std(axis=0) * sigma2_outputs.std(axis=0) + 1e-5)
                    print('Output uncertainty corr w. error:', pearson_corr.mean())
                    print('Mutual Information err2 with sigma2: ', compute_mi(err2.detach().cpu().numpy(), sigma2_outputs.detach().cpu().numpy()), '/', compute_mi(err2.detach().cpu().numpy(), err2.detach().cpu().numpy()))
                        # self._vis_loss_surface(self.model,
                        #                     self.maxlikelihood_criterion, 
                        #                     data, 
                        #                     'model.reconstruct_muy_A.net.0.weight', 
                        #                     'model.reconstruct_muy_A.net.0.bias',
                        #                     delta=4, steps=100)
                elif self.dataset in ['mmimdb', 'food101', 'hatememes']:
                    if self.labeled_ratio == 0.0:
                        train_reps = []
                        train_truths = []

                        for i_batch, (batch_X, batch_Y) in tqdm(enumerate(self.train_loader)):
                            image, text = batch_X
                            data = [text.to(self.device), image.to(self.device)]
                            target_data, _ = batch_Y
                            target_data = target_data.float().squeeze(-1).to(self.device)
                            
                            # Drop modalities (if required)
                            input_data = []
                            if k < self.test_modalities:
                                # single modality testing
                                for j in range(len(data)):
                                    if j != k:
                                        input_data.append(None)
                                    else:
                                        input_data.append(data[j])
                            else:
                                # joint modality testing
                                input_data = data

                            rep = self.model.encode(input_data, return_reps=True)
                            train_reps.append(rep.detach().cpu().numpy())
                            train_truths.append(target_data.detach().cpu().numpy())
                        
                        # train classifier
                        train_reps = np.concatenate(train_reps, axis=0)
                        train_truths = np.concatenate(train_truths, axis=0)
                        classifier = fit_lr(train_reps, train_truths.squeeze(-1))             
                                 
                    results = []
                    truths = []
                    for i_batch, (batch_X, batch_Y) in tqdm(enumerate(self.test_loader)):
                        image, text = batch_X
                        data = [text.to(self.device), image.to(self.device)]
                        target_data, _ = batch_Y
                        target_data = target_data.float().squeeze(-1).to(self.device)  # if num of labels is 1

                        # Drop modalities (if required)
                        input_data = []
                        if k < self.test_modalities:
                            # single modality testing
                            for j in range(len(data)):
                                if j != k:
                                    input_data.append(None)
                                else:
                                    input_data.append(data[j])
                        else:
                            # joint modality testing
                            input_data = data

                        # Parallel model
                        if self.labeled_ratio != 0.0:
                            preds = self.model.encode(input_data)
                        else:
                            rep = self.model.encode(input_data, return_reps=True)
                            preds = torch.from_numpy(classifier.predict(rep.cpu().detach().numpy()))

                        # Collect the results into dictionary
                        truths.append(target_data)
                        results.append(preds)

                    results = torch.cat(results)
                    truths = torch.cat(truths)

                    if self.dataset == "mmimdb":
                        calculate_f1(results, truths, self.logger)
                    elif self.dataset == 'food101':
                        calculate_accuracy(results, truths, self.logger)
                    elif self.dataset == 'hatememes':
                        calculate_auroc(results, truths, self.logger)

    def _vis_sigma(self, reps, muys, sigma2s):
        n_mod = len(reps)
        fig, axs = plt.subplots(n_mod, 3, figsize=(20, 9))
        for mod in range(n_mod):
            reps0 = reps[mod].detach().cpu().numpy().mean(axis=1)#[:50] # representation (B, T, H)
            muys0 = muys[mod].detach().cpu().numpy().mean(axis=1)#[:50]
            # sigma2s0 = sigma2s[mod].detach().cpu().numpy().mean(axis=1)#[:50] #[:, 5, :] # (B, T, H) -> (B,)
            sigma2s0 = sigma2s[mod].detach().cpu().numpy().mean(axis=-1)#[:50] #[:, 5, :] # (B, T, H) -> (B,)

            err2 = (reps0 - muys0)**2
            cov = (err2 - err2.mean(axis=0, keepdims=True)) * (sigma2s0 - sigma2s0.mean(axis=0, keepdims=True))
            pearson_corr = cov.mean(axis=0) / (err2.std(axis=0) * sigma2s0.std(axis=0) + 1e-5)
            print(10*'*' + 'Pearson Correlation: ', pearson_corr.mean())
            xs = range(reps0.shape[0])
            # plot the variance of the reconstruction over training process
            axs[mod][0].plot(xs, muys0.mean(axis=-1), 'bo')
            axs[mod][0].errorbar(xs, muys0.mean(axis=-1), yerr=sigma2s0.squeeze(), fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0)
            axs[mod][0].plot(xs, reps0.mean(axis=-1), 'ro')
            
            axs[mod][0].set_xlabel('Samples')
            axs[mod][0].set_ylabel('GT vs Muy/Sigma2')
            axs[mod][0].set_title('Reconstruction vs Ground Truth')
            
            idx = np.argsort(sigma2s0, axis=0)
            axs[mod][1].bar(xs, sigma2s0[idx].squeeze(), color='tab:blue', alpha=0.5, label='sigma')
            axs[mod][1].bar(xs, err2[idx].mean(axis=-1).squeeze(), color='tab:red', alpha=0.7, label='err')
            axs[mod][1].set_xlabel('Samples')
            axs[mod][1].set_ylabel('Err2 and Sigma2')
            axs[mod][1].legend(loc='best')

            axs[mod][2].hist(sigma2s0.squeeze(), bins=50, density=True)
            axs[mod][2].set_xlabel('Bin')
            axs[mod][2].set_ylabel('Density')

        fig.tight_layout()
        plt.savefig(f'logs/{self.dataset}/reconstruction.png', dpi=300, bbox_inches='tight')
    
    def _vis_sigma_multimodal(self, reps, muys, sigma2s):
        n_mod = self.test_modalities
        fig, axs = plt.subplots(n_mod, 3, figsize=(20, 9))
        for mod in range(n_mod):
            reps_i = torch.cat([r[mod] for r in reps], dim=0).detach().cpu().numpy()#[:50] # representation (B, T, H)
            
            muys_i = [muy[mod] for muy in muys]
            muys_is = [torch.cat([muy[k] for muy in muys_i], dim=0) for k in muys_i[0].keys()]
            
            sigma2s_i = [sigma2[mod] for sigma2 in sigma2s]
            sigma2s_is = [torch.cat([sigma2[k] for sigma2 in sigma2s_i], dim=0) for k in sigma2s_i[0].keys()]
            
            for k in range(len(muys_is)):
                muys0 = muys_is[k].detach().cpu().numpy()
                sigma2s0 = sigma2s_is[k].detach().cpu().numpy()
                err2 = ((reps_i - muys0)**2)#.mean(axis=-1)
                err2 = np.linalg.norm(err2, axis=-1)
                sigma2s0 = np.linalg.norm(sigma2s0, axis=-1)
                print(err2.shape, sigma2s0.shape)
                cov = (err2 - err2.mean(axis=0, keepdims=True)) * (sigma2s0 - sigma2s0.mean(axis=0, keepdims=True))
                pearson_corr = cov.mean(axis=0) / (err2.std(axis=0) * sigma2s0.std(axis=0) + 1e-5)
                # print(10*'*' + 'Mod: ', mod, ', Pearson Correlation: ', pearson_corr.mean(), 'Reconstruction Error: ', np.linalg.norm(err2, axis=-1).mean())
                print(10*'*' + 'Mod: ', mod, ', Pearson Correlation: ', pearson_corr.mean(), 'Reconstruction Error: ', err2.mean())
                print(10*'*' + 'Mod: ', mod, ', Mutual Information err2: ', compute_mi(err2, sigma2s0), '/', compute_mi(err2, err2))
                # err2 = err2.mean(axis=-1)
                # sigma2s0 = sigma2s0.mean(axis=-1)
                # import IPython; IPython.embed(); exit()
                # plot the variance of the reconstruction over training process
                if k == 0:
                    xs = range(reps_i.shape[0])
                    axs[mod][0].plot(xs, muys0.mean(axis=-1), 'bo')
                    axs[mod][0].errorbar(xs, muys0.mean(axis=-1), yerr=sigma2s0, fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0)
                    axs[mod][0].plot(xs, reps_i.mean(axis=-1), 'ro')
                    
                    axs[mod][0].set_xlabel('Samples')
                    axs[mod][0].set_ylabel('GT vs Muy/Sigma2')
                    axs[mod][0].set_title('Reconstruction vs Ground Truth')
            
                    idx = np.argsort(sigma2s0, axis=0)
                    axs[mod][1].bar(xs, sigma2s0[idx], color='tab:blue', alpha=0.5, label='sigma')
                    axs[mod][1].bar(xs, err2[idx], color='tab:red', alpha=0.7, label='err')
                    axs[mod][1].set_xlabel('Samples')
                    axs[mod][1].set_ylabel('Err2 and Sigma2')
                    axs[mod][1].legend(loc='best')

                    axs[mod][2].hist(sigma2s0, bins=50, density=True)
                    axs[mod][2].set_xlabel('Bin')
                    axs[mod][2].set_ylabel('Density')

        fig.tight_layout()
        plt.savefig(f'logs/{self.dataset}/reconstruction.png', dpi=300, bbox_inches='tight')

    def _vis_sigma_single_dim(self, reps, muys, sigma2s):
        n_mod = len(reps)
        fig, axs = plt.subplots(n_mod, 2, figsize=(20, 9))
        for mod in range(n_mod):
            reps0 = reps[mod].detach().cpu().numpy()#[:50] # representation (B, T, H)
            muys0 = muys[mod].detach().cpu().numpy()#[:50]
            sigma2s0 = sigma2s[mod].detach().cpu().squeeze().numpy()#[:50] #[:, 5, :] # (B, T, H) -> (B,)
            sigma2s0 = np.linalg.norm(sigma2s0, axis=-1)

            err2 = ((reps0 - muys0)**2)#.mean(axis=-1)
            err2 = np.linalg.norm(err2, axis=-1)
            cov = (err2 - err2.mean(axis=0, keepdims=True)) * (sigma2s0 - sigma2s0.mean(axis=0, keepdims=True))
            pearson_corr = cov.mean(axis=0) / (err2.std(axis=0) * sigma2s0.std(axis=0) + 1e-5)
            # print(10*'*' + 'Mod: ', mod, ', Pearson Correlation: ', pearson_corr.mean(), 'Reconstruction Error: ', np.linalg.norm(err2, axis=-1).mean())
            print(10*'*' + 'Mod: ', mod, ', Pearson Correlation: ', pearson_corr.mean(), 'Reconstruction Error: ', err2.mean())
            print(10*'*' + 'Mod: ', mod, ', Mutual Information err2: ', compute_mi(err2, sigma2s0), '/', compute_mi(err2, err2))
            # err2 = err2.mean(axis=-1)
            # sigma2s0 = sigma2s0.mean(axis=-1)
            # import IPython; IPython.embed(); exit()
            # plot the variance of the reconstruction over training process
            xs = range(reps0.shape[0])
            idx = np.argsort(sigma2s0, axis=0)
            axs[mod][0].bar(xs, sigma2s0[idx], color='tab:blue', alpha=0.5, label='sigma')
            axs[mod][0].set_xlabel('Samples')
            axs[mod][0].set_ylabel('Err2 and Sigma2')
            
            ax2 = axs[mod][0].twinx()
            ax2.bar(xs, err2[idx], color='tab:red', alpha=0.7, label='err')
            axs[mod][0].legend(loc='upper left')

            axs[mod][1].hist(sigma2s0, bins=50, density=True)
            axs[mod][1].set_xlabel('Bin')
            axs[mod][1].set_ylabel('Density')

        fig.tight_layout()
        plt.savefig(f'logs/{self.dataset}/reconstruction.png', dpi=300, bbox_inches='tight')

    def _vis_loss_surface(self, model, criterion, X, param1, param2, delta=1, steps=1000):
        # losses = np.zeros((steps, steps))
        model.eval()
        
        # compute the loss surface
        def tau_2d(alpha, beta, theta_ast):
            a = alpha * theta_ast[:,None,None]
            b = beta * alpha * theta_ast[:,None,None]
            return a + b

        x = torch.linspace(-1000, 1000, 10)
        y = torch.linspace(-1000, 1000, 10)
        alpha, beta = torch.meshgrid(x, y)
        space = tau_2d(alpha, beta, self.theta_ast)

        losses = torch.empty_like(space[0, :, :])
        X = [a.to('cpu') for a in X]
        for a, _ in enumerate(x):
            print(f'a = {a}')
            for b, _ in enumerate(y):
                Vec2Params(space[:, a, b], self.model.parameters())
                with torch.no_grad():
                    self.model.eval()
                    _, real_outputs, mean_outputs, variance_outputs = self.model.encode(X)
                    loss = criterion(real_outputs[1], mean_outputs[1], variance_outputs[1])
                    losses[a][b] = loss.item()

        # visualize
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(np.array(w1_changes), np.array(w2_changes), losses, cmap='viridis') #, rstride=1, cstride=1, cmap='viridis')
        ax.plot_surface(alpha, beta, losses, cmap='viridis') #, rstride=1, cstride=1, cmap='viridis')
        _, real_outputs, mean_outputs, variance_outputs = model.encode(X)
        loss = criterion(real_outputs[1], mean_outputs[1], variance_outputs[1]).item()
        # point_x = model.state_dict()[param1][0].mean().item()
        # point_y = model.state_dict()[param2][0].mean().item()
        # ax.scatter(point_x, point_y, loss, color='red', s=5, alpha=1)
        ax.set_xlabel('Perturbation of ' + param1)
        ax.set_ylabel('Perturbation of ' + param2)
        ax.set_zlabel('Loss')
        plt.title('Loss Landscape')

        fig.tight_layout()
        plt.savefig(f'logs/{self.dataset}/loss_surface.png', dpi=300, bbox_inches='tight')