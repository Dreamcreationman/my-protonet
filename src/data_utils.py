import torch
import numpy as np


class BatchSampler(object):

    def __init__(self, labels, classes_per_eps, num_samples, episodes):
        """

        :param classes_per_iter:
        :param num_samples:
        :param episodes:
        """
        super(BatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_eps = classes_per_eps
        self.sampler_per_class = num_samples
        self.episodes = episodes

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.labels))
        self.indexes = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        cpe = self.classes_per_eps
        spc = self.sampler_per_class

        for epi in range(self.episodes):
            batch_size = cpe * spc
            batch = torch.LongTensor(batch_size)
            classes_idx = torch.randperm(len(self.classes))[:cpe]
            for i, c in enumerate(self.classes[classes_idx]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.episodes
