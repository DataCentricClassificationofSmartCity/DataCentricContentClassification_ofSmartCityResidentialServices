import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

RANDOM_SEED = 26
PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


# field = torchtext.data.Field(
#         tokenize=str.split,
#         lower=True,
#         pad_token=PAD_WORD,
#         init_token=BOS_WORD,
#         eos_token=EOS_WORD)

# fields = (field, field)

def padding_data(data, max_length, padding_value):
    '''
        padding a list of lists to max_length, and return a 2D list
        Arguments:
            data {Pandas Series} -- sequence to be padded, each element is a list of word index
            max_length {Int} -- the max length of the sequences
            padding_value {Int} -- the value to fill in
        Returns:
            data {2-D Numpy ndarray} -- a 2-D array with shape [number of sequences, max_length]
    '''

    def padding(example):
        '''padding the lists to the same length'''
        seq_length = len(example)
        if seq_length <= max_length:
            return np.pad(example, (0, max_length - seq_length), constant_values=padding_value, mode='constant')
        else:
            example = example[:max_length]
            return example

    data = data.map(padding)
    data = np.concatenate(data.values)
    data = data.reshape(-1, max_length)  # Squeeze the lists to one list, and reshape

    return data


def process_indexes(index_series, max_length=None):
    '''
        Argument:
            index_series {Pandas Series} -- A Series of indexes, each element is a string contains the indexes
        Returns:
            idx_seq {Numpy ndarray, shape [n_rows_of_data, input_length]} -- a 2-D array of word indexes, padding with 0
            pos_seq {Numpy ndarray, shape [n_rows_of_data, input_length]} -- a 2-D array of position indexes, padding with 0
            input_length {Int} -- the max length of all lists in the Series
    '''
    # string to list
    # index_series = index_series.reset_index()

    indexes = index_series.map(lambda x: eval(x))
    # get list length
    length = indexes.map(lambda x: len(x))
    # generate position
    position = length.map(lambda x: list(range(0, x)))
    # get the max length
    max_length_ = length.max()
    # indexes {Pandas Series} -- each element is a list of indexes
    # length {Pandas Seires} -- each element is the length of the list of indexes
    # position {Pandas Series} -- each element is a list of positions
    # input_length {Int} -- the max length of all lists in the Series

    if max_length is None:
        max_length = max_length_
    else:
        if max_length < max_length_:
            print(
                f'Exists index sequence with a length larger than input_length, cut to {max_length}.')

    # Pad the sequences with -1 to the same length (input_length), for the use of indexing, shift the sequence with 1
    idx_seq = padding_data(indexes, max_length, padding_value=-1) + 1
    pos_seq = padding_data(position, max_length, padding_value=-1) + 1

    return idx_seq, pos_seq, max_length


class SeqLabelDataset(Dataset):
    def __init__(self, dataframe, cut_length=None, seq_length=100, out_length=None):
        super().__init__()
        # Load training data

        self.input_indexes, self.input_position_indexes, input_length = process_indexes(dataframe['input_indexes'],
                                                                                        seq_length)

        self.target_indexes, self.output_position_indexes, output_length = process_indexes(dataframe['target_indexes'],
                                                                                           out_length)
        self.classes = dataframe['class'].tolist()

        self.input_length = input_length

        self.output_length = output_length

        if not (cut_length is None):
            self.input_indexes = self.input_indexes[:, :cut_length]
            self.input_position_indexes = self.input_position_indexes[:, :cut_length]
            self.input_length = cut_length

    def __len__(self):
        return len(self.target_indexes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.input_indexes[index, :].astype(np.long)),
                  torch.from_numpy(self.input_position_indexes[index, :].astype(np.long)),
                  torch.from_numpy(self.target_indexes[index, :].astype(np.long)),
                  torch.from_numpy(self.output_position_indexes[index, :].astype(np.long)),
                  self.classes[index]
                  )

        return sample


class SeqLabelDataloader():
    def __init__(self, train_data, test_data, val_data, batch_size, cut_length=None, in_len=None, out_len=None,
                 test=False):
        self.cut_length = cut_length

        if out_len is not None:
            self.output_length = out_len

        if in_len is not None:
            self.max_length = in_len

        if not test:
            self.train_loader, in_length, out_length = self.init_dataloader(train_data, batch_size)
            self.val_loader = self.init_dataloader(val_data, batch_size, out_length)
            self.test_loader = None
            self.max_length = in_length
            self.output_length = out_length

        else:
            self.train_loader = None
            self.val_loader = None
            self.test_loader= self.init_dataloader(test_data, 1, self.output_length)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def init_dataloader(self, data, batch_size=128, out_len=None):
        if out_len is None:
            dataset = SeqLabelDataset(data, cut_length=self.cut_length)
            output_length = dataset.output_length
            input_length = dataset.input_length
        else:
            dataset = SeqLabelDataset(data, cut_length=self.cut_length, out_length=out_len)
        data_size = len(dataset)
        indices = list(range(data_size))
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset, batch_size, sampler=sampler)
        if out_len is None:
            return dataloader, input_length, output_length
        else:
            return dataloader
