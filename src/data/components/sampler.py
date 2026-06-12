import torch
from torch.utils.data import Sampler, DataLoader, Dataset
import numpy as np
import random

class SortedBatchSampler(Sampler):
    def __init__(self, indicator, batch_size=32, shuffle=False):
        super().__init__()
        self.indicator = indicator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = []
        self.setup()
    
    def setup(self):
        indexes = torch.argsort(self.indicator).detach().cpu().tolist()
        self.data = []

        batch_idxes = range(0, len(indexes), self.batch_size)
        # Append till sub-last batch
        batches = [indexes[idx:idx+self.batch_size] for idx in batch_idxes[:-2]]
        # Last batch random pick batch_size samples
        last_batch = indexes[batch_idxes[-2]:]
        last_batch_idxs = np.random.choice(last_batch, self.batch_size)
        batches.append(last_batch_idxs)
        self.data = batches
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        batches = list(self.data)
        if self.shuffle:
            random.shuffle(batches)
        # Yield one batch of indices at a time
        for batch in batches:
            yield batch
