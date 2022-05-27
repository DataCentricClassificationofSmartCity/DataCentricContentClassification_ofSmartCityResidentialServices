import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from src.training.transformer_utils import process_indexes

RANDOM_SEED = 26


class TextualDataset(Dataset):
    def __init__(self, dataset, n_classes=73,  cut_length=100):
        super().__init__()
        # Load training data

        self.input_sequence_index, self.position_index, max_length = process_indexes(dataset['input_indexes'])

        self.targets = dataset['class'].values.reshape(-1, 1)

        self.n_classes = n_classes

        self.max_length = max_length

        if not (cut_length is None):
            self.input_sequence_index = self.input_sequence_index[:, :cut_length]
            self.position_index = self.position_index[:, :cut_length]
            self.max_length = cut_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.input_sequence_index[index, :].astype(np.long)),
                  torch.from_numpy(self.position_index[index, :].astype(np.long)),
                  torch.from_numpy(self.targets[index, :]))

        return sample


class TextualDataloader():
    def __init__(self, train_path, test_path, batch_size, eval_portion, cut_length=None, shuffle=True):
        train_set = TextualDataset(train_path, cut_length=cut_length)
        test_set = TextualDataset(test_path, cut_length=cut_length)

        self.max_length = train_set.max_length

        self.n_targets = train_set.n_classes

        train_size = len(train_set)
        train_indices = list(range(train_size))

        test_size = len(test_set)
        test_indices = list(range(test_size))
        split = int(np.floor(eval_portion * test_size))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(test_indices)

        test_indices, eval_indices = test_indices[split:], test_indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        # train_sampler = BalanceSampler(train_set.targets, train_indices)
        valid_sampler = SubsetRandomSampler(eval_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = DataLoader(train_set, batch_size, sampler=train_sampler, num_workers=4)
        self.val_loader = DataLoader(test_set, batch_size, sampler=valid_sampler, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size, sampler=test_sampler, num_workers=4)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class TextualDataloaderWithSplitEval():
    def __init__(self, train_dataset, test_dataset, val_dataset, batch_size, cut_length=100):
        self.train_set = TextualDataset(train_dataset, cut_length=cut_length)
        self.test_set = TextualDataset(test_dataset, cut_length=cut_length)
        self.eval_set = TextualDataset(val_dataset, cut_length=cut_length)

        self.max_length = self.train_set.max_length

        self.n_classes = self.train_set.n_classes

        train_size = len(self.train_set)
        train_indices = list(range(train_size))

        train_sampler = SubsetRandomSampler(train_indices)
        # train_sampler = BalanceSampler(self.train_set.targets, train_indices)

        self.train_loader = DataLoader(self.train_set, batch_size, sampler=train_sampler, num_workers=4)
        self.val_loader = DataLoader(self.eval_set, batch_size, num_workers=4)
        self.test_loader = DataLoader(self.test_set, batch_size, num_workers=4)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
