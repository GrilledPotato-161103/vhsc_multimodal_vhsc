import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON=1e-10
EPSILON2=1e-4

class NCELoss(torch.nn.Module):
    def __init__(self, dataset='mosi'):
        super(NCELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dataset = dataset


    def where(self, cond, x_1, x_2):
        cond = cond.type(torch.float32)
        return (cond * x_1) + ((1 - cond) * x_2)

    def _discretize(self, x):
        return x.round(decimals=1)
       
    def forward(self, f1, f2, targets, predictions, target_mask, temperature=0.3):
        ### cuda implementation

        if self.dataset in ['mosi', 'mosei']:
            ## discretize the predictions and targets
            targets = self._discretize(targets)
            predictions = self._discretize(predictions)

        ## set distances of the same label to zeros
        padded_targets = torch.zeros_like(targets)
        padded_targets[target_mask] = targets[target_mask]
        padded_targets[~target_mask] = predictions[~target_mask]
        # padded_targets = targets
        mask = padded_targets.unsqueeze(1) - padded_targets
        self_mask = (torch.zeros_like(mask) != mask).squeeze()  
        ### where the negative samples are labeled as True
        
        cos = torch.mm(f1, f2.t().contiguous())

        ## convert cos distance to exponential space
        pred_softmax = self.softmax(cos / temperature) ### convert to multi-class prediction scores

        log_pos_softmax = - torch.log(pred_softmax + EPSILON) * (1 - self_mask.float())
        log_neg_softmax = - torch.log(1 - pred_softmax + EPSILON) * self_mask.float()
        log_softmax = log_pos_softmax.sum(1) / (1 - self_mask.float()).sum(1) + log_neg_softmax.sum(1) / self_mask.sum(1).float()
        # loss = log_softmax

        if torch.isnan(log_softmax).any():
            import IPython; IPython.embed(); exit(1)

        return log_softmax

class GaussianAlignLoss(torch.nn.Module):
    def __init__(self, dataset='mosi', gamma=1, task='regression'):
        super(GaussianAlignLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dataset = dataset
        self.resi_min = 1e-4
        self.resi_max = 1e2
        self.log_min = -1e0
        self.log_max = 1e1
        self.gamma = gamma
        self.task = task
        print('gamma: ', gamma)

    def forward_bak(self, f, muy, sigma2, mod, beta=1):
        f = f.detach()
        resi = (f - muy)**2
        resi = resi.mean(dim=-1, keepdim=True)
        # resi_ = (resi - resi.min(dim=-1, keepdim=True)[0])/(resi.max(dim=-1, keepdim=True)[0] - resi.min(dim=-1, keepdim=True)[0] + EPSILON) + 1
        # sigma2 = (sigma2 - sigma2.min(dim=-1, keepdim=True)[0])/(sigma2.max(dim=-1, keepdim=True)[0] - sigma2.min(dim=-1, keepdim=True)[0] + EPSILON) + 1

        resi_loss = (resi / (sigma2 + EPSILON))#.clamp(min=self.resi_min, max=self.resi_max)
        log_sigma = (torch.log(sigma2 + EPSILON))#.clamp(min=self.log_min, max=self.resi_max)
        loss = resi_loss + log_sigma
        # print('err2 norm:', resi_loss.norm(), 'sigma2 norm: ', sigma2.norm())
        cov = (resi - resi.mean(axis=0, keepdims=True)) * (sigma2 - sigma2.mean(axis=0, keepdims=True))
        pearson_corr = cov.mean(axis=0) / (resi.std(axis=0, unbiased=False) * sigma2.std(axis=0, unbiased=False) + EPSILON)

        corr_loss = (1 - pearson_corr).abs().mean()

        if torch.isnan(loss).any():
            import IPython; IPython.embed(); exit(0)
            
        return beta * torch.mean(loss) + self.gamma * corr_loss
        # return self.gamma * corr_loss + resi.mean()
        # return loss.mean()
        # return resi.mean()

    def forward(self, f, muy, sigma2, mod, beta=1):
        f = f.detach()
        if self.task == 'regression':
            resi = (f - muy)**2
            resi = resi.norm(p=2, dim=-1)
            # resi_ = resi.mean(dim=-1, keepdim=True)
            sigma2 = sigma2.norm(p=2, dim=-1)
        elif self.task == 'classification':
            resi = F.cross_entropy(muy, f, reduction='none').unsqueeze(-1)
        
        cov = (resi - resi.mean(axis=0, keepdims=True)) * (sigma2 - sigma2.mean(axis=0, keepdims=True))
        pearson_corr = cov.mean(axis=0) / (resi.std(axis=0, unbiased=False) * sigma2.std(axis=0, unbiased=False) + EPSILON)

        corr_loss = (1 - pearson_corr).abs().mean()
        # corr_loss = (resi_ - sigma2)**2
        # print('corr loss: ', corr_loss.mean().item(), 'resi loss: ', resi.norm(p=2, dim=-1).mean().item(), 'beta: ', beta, 'gamma: ', self.gamma)
        
        if torch.isnan(corr_loss).any():
            import IPython; IPython.embed(); exit(0)
            
        return beta * resi.mean() + self.gamma * corr_loss.mean()

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
       
    def forward(self, pred, gtruth, masks, sigma2=None, temperature=0.3):
        ### cuda implementation
        total_mask = torch.logical_and(masks[:, 0], masks[:, 1])
        if sigma2 is None:
            weight = torch.zeros_like(gtruth, requires_grad=False)
            weight[total_mask] = 1
        else:
            sigma2 = [sigma2[0].mean(dim=1).mean(1), sigma2[1].mean(dim=1).mean(1)]
            sigma2_unique = torch.logical_xor(masks[:, 0], total_mask) * sigma2[0] + torch.logical_xor(masks[:, 1], total_mask) * sigma2[1]
            
            weight = torch.zeros_like(gtruth, dtype=torch.float32)
            incomplete_mask = torch.logical_xor(masks[:, 0], masks[:, 1])
            weight[incomplete_mask] = sigma2_unique[incomplete_mask].min() / sigma2_unique[incomplete_mask]
            weight[total_mask] = 1

        weight = weight.detach()
        # weighted cross entropy loss, with weight for samples with incomplete labels
        loss = F.cross_entropy(pred, gtruth, reduction='none')
        if torch.isnan(loss).any():
            import IPython; IPython.embed(); exit(1)
            
        loss = (loss * weight).mean()
        return loss
        
class WeightedL1Loss(torch.nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()
       
    def forward(self, pred, gtruth, masks, sigma2=None, temperature=0.3):
        ### cuda implementation
        total_mask = torch.logical_and(masks[:, 0], masks[:, 1])
        if sigma2 is None:
            weight = torch.zeros_like(gtruth, requires_grad=False)
            weight[total_mask] = 1
        else:
            sigma2 = [sigma2[0].mean(1), sigma2[1].mean(1)]
            sigma2_unique = torch.logical_xor(masks[:, 0], total_mask) * sigma2[0] + torch.logical_xor(masks[:, 1], total_mask) * sigma2[1]
            
            weight = torch.zeros_like(gtruth, dtype=torch.float32)
            incomplete_mask = torch.logical_xor(masks[:, 0], masks[:, 1]).squeeze()
            if torch.any(incomplete_mask):
                weight[incomplete_mask] = (sigma2_unique[incomplete_mask].min() / sigma2_unique[incomplete_mask]).unsqueeze(-1)
            
            weight[total_mask] = 1

        weight = weight.detach()
        # weighted cross entropy loss, with weight for samples with incomplete labels
        # loss = F.cross_entropy(pred, gtruth, reduction='none')
        loss = F.l1_loss(pred, gtruth, reduction='none')
        if torch.isnan(loss).any():
            import IPython; IPython.embed(); exit(1)
            
        loss = (loss * weight).mean()
        return loss

class OrderedEnforceLoss(torch.nn.Module):
    def __init__(self, dataset='mosi'):
        super(OrderedEnforceLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dataset = dataset
    
    def forward(self, f, muy, sigma2, y, temperature=0.3):
        ### cuda implementation
        # clip sigma2 to avoid numerical instability
        clip_mask = sigma2 < EPSILON2
        sigma2[clip_mask] = torch.clamp(sigma2[clip_mask], min=EPSILON2)

        mask = (y.unsqueeze(1) - y).detach()
        positive_mask = (mask == 0).unsqueeze(-1).unsqueeze(-1)
        negative_mask = (mask != 0).unsqueeze(-1).unsqueeze(-1)

        f_rep = f.unsqueeze(0).repeat(f.size(0), 1, 1, 1)
        muy_rep = muy.unsqueeze(1).repeat(1, muy.size(0), 1, 1)
        sigma2_rep = sigma2.unsqueeze(1).repeat(1, sigma2.size(0), 1, 1)
        likelihood = (f_rep - muy_rep)**2 / (2 * sigma2_rep + EPSILON) + torch.log(torch.sqrt(sigma2 * 2 * torch.pi) + EPSILON)
        positive_likelihood = likelihood * positive_mask
        negative_likelihood = likelihood * negative_mask

        loss = positive_likelihood.sum(dim=0) / (positive_mask.sum(0) + EPSILON) - negative_likelihood.sum(dim=0) / (negative_mask.sum(0) + EPSILON)
        
        if torch.isnan(loss).any():
            import IPython; IPython.embed(); exit(1)

        return torch.mean(loss)

if __name__ == '__main__':
    loss = OrderedEnforceLoss()
    f = torch.randn(32, 5, 768)
    muy = torch.randn(32, 5, 768)
    sigma = F.normalize(torch.randn(32, 5, 768) ** 2)
    targets = torch.randint(0, 33, (32,))

    print(loss(f, muy, sigma, targets))