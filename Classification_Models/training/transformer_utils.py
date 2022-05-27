import gensim
import numpy as np
import torch
import torch.nn.functional as F
import enum


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
        return np.pad(example, (0, max_length - seq_length), constant_values=padding_value, mode='constant')

    data = data.map(padding)
    data = np.concatenate(data.values)
    data = data.reshape(-1, max_length)  # Squeeze the lists to one list, and reshape

    return data


def process_indexes(index_series, max_length=None):
    '''
        Argument:
            index_series {Pandas Series} -- A Series of indexes, each element is a string contains the indexes
        Returns:
            idx_seq {Numpy ndarray, shape [n_rows_of_data, max_length]} -- a 2-D array of word indexes, padding with 0
            pos_seq {Numpy ndarray, shape [n_rows_of_data, max_length]} -- a 2-D array of position indexes, padding with 0
            max_length {Int} -- the max length of all lists in the Series
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
    # max_length {Int} -- the max length of all lists in the Series

    if max_length is None:
        max_length = max_length_
    else:
        if max_length < max_length_:
            raise Exception(
                'Exists index sequence with a length larger than max_length, please check if max_length applies for all of your datasets.')

    # Pad the sequences with -1 to the same length (max_length), for the use of indexing, shift the sequence with 1
    idx_seq = padding_data(indexes, max_length, padding_value=-1) + 1
    pos_seq = padding_data(position, max_length, padding_value=-1) + 1

    return idx_seq, pos_seq, max_length


def one_hot_embedding(index, num_classes):
    diag = torch.eye(num_classes).byte()
    index = index.view(-1).tolist()
    return diag[index]


def get_embeddings(model_path, padding=False):
    '''
        Arguments:
            model_path {String} -- gensim model file path
            padding {Boolean} -- Whether to expand a 0 vector for padding
        Returns:
            embedding {Tensor, shape [n_vocabulary, embedding length]} -- word embedding vectors
            vector_length {Int} -- embedding length
    '''
    # Load word2vec model
    word2vec = gensim.models.Word2Vec.load(model_path)
    vector_length = word2vec.vector_size
    embedding = torch.FloatTensor(word2vec.wv.vectors)
    # Leave the first embedding to 0 for padding
    if padding:
        empty_embedding = torch.zeros_like(embedding[0]).unsqueeze(0)
        embedding = torch.cat([empty_embedding, embedding])

    return embedding, vector_length


def accuracy(pred, real, threshold=None, ignore_idx=None):
    if ignore_idx is not None:
        mask = real != ignore_idx
        pred = pred[mask]
        real = real[mask]
    if threshold == None:
        n = real.shape[0]
        pred = pred.argmax(dim=-1).view(n, -1)
        real = real.view(n, -1)
        acc = pred.byte() == real.byte()
        acc = (acc).float().mean()
        return acc
    else:
        acc = (pred > threshold).byte() == real.byte()
        acc = (acc).float().mean()
        return acc


def precision_recall(pred, real, d_output, threshold=None, average='macro', eps=1e-6):
    n = real.shape[0]
    dim = real.shape[-1]
    if dim == 1:
        if threshold != None:
            pred = (pred > threshold).byte()
            pred = one_hot_embedding(pred, 2)
            real = one_hot_embedding(real, 2)
        else:
            pred = one_hot_embedding(torch.argmax(pred, -1).view(n, -1), d_output)
            real = one_hot_embedding(real, d_output)

    tp = (pred & real)
    tp_count = tp.sum(0).float()
    fp_count = (pred - tp).sum(0).float() + eps
    fn_count = (real - tp).sum(0).float() + eps
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)

    if threshold != None:
        return precision, recall, precision, recall

    else:
        if average == 'macro':
            precision_avg = precision.mean()
            recall_avg = precision.mean()

        else:
            precision_avg = tp.sum(0).sum() / n
            recall_avg = precision_avg

    return precision, recall, precision_avg, recall_avg


class Evaluator():
    def __init__(self):
        self.total_loss = 0
        self.total_acc = 0
        self.total_pre = 0
        self.total_rec = 0
        self.batch_counter = 0

    def __call__(self, loss, acc, precision, recall):
        self.total_loss += loss
        self.total_acc += acc
        self.total_pre += precision
        self.total_rec += recall
        self.batch_counter += 1

    def avg_results(self):
        loss_avg = self.total_loss / self.batch_counter
        acc_avg = self.total_acc / self.batch_counter
        pre_avg = self.total_pre / self.batch_counter
        rec_avg = self.total_rec / self.batch_counter
        return loss_avg, acc_avg, pre_avg, rec_avg


def cross_entropy_loss(logits, real, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    real = real.contiguous().view(-1).long()

    if smoothing:
        eps = 0.1
        n_class = logits.size(1)
        pred = logits.log_softmax(dim=1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(eps / (n_class - 1))
            true_dist.scatter_(1, real.data.unsqueeze(1), 1 - eps)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))
    else:
        # loss = F.cross_entropy(logits, real, ignore_index=0, reduction='sum')
        loss = F.cross_entropy(logits, real)
        return loss


def mse_loss(pred, real):
    loss = F.mse_loss(pred, real)

    return loss


class TrainingControl():
    def __init__(self, max_step, evaluate_every_nstep, print_every_nstep):
        self.state_dict = {
            'epoch': 0,
            'batch': 0,
            'step': 0,
            'step_to_evaluate': False,
            'step_to_print': False,
            'step_to_stop': False
        }
        self.max_step = max_step
        self.eval_every_n = evaluate_every_nstep
        self.print_every_n = print_every_nstep
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0

    def __call__(self, batch):
        self.current_step += 1
        self.state_dict['batch'] = batch
        self.state_dict['step'] = self.current_step
        self.state_dict['step_to_evaluate'] = np.equal(np.mod(self.current_step, self.eval_every_n), 0)
        self.state_dict['step_to_print'] = np.equal(np.mod(self.current_step, self.print_every_n), 0)
        self.state_dict['step_to_stop'] = np.equal(self.current_step, self.max_step)
        return self.state_dict

    def set_epoch(self, epoch):
        self.state_dict['epoch'] = epoch

    def reset_state(self):
        self.state_dict = {
            'epoch': 0,
            'batch': 0,
            'step': 0,
            'step_to_evaluate': False,
            'step_to_print': False,
            'step_to_stop': False
        }
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0


class MetricsEnum():
    ACC = 'acc'
    LOSS = 'loss'


class EarlyStopping:
    def __init__(self, patience, mode='best', on='acc'):
        self.patience = patience
        self.mode = mode
        self.best_metrics = 0
        self.waitting = 0
        self.on_metrics = on
        self.state_dict = {
            'save': False,
            'break': False
        }
        if on == MetricsEnum.LOSS:
            self.best_metrics = 999999

    def __call__(self, m):
        self.state_dict['save'] = False
        self.state_dict['break'] = False

        if ((self.on_metrics == MetricsEnum.ACC) & (m >= self.best_metrics)) or (
                (self.on_metrics == MetricsEnum.LOSS) & (m < self.best_metrics)):
            self.best_metrics = m
            self.waitting = 0
            self.state_dict['save'] = True
        else:
            self.waitting += 1

            if self.mode == 'best':
                self.state_dict['save'] = False
            else:
                self.state_dict['save'] = True

            if self.waitting >= self.patience:
                self.state_dict['break'] = True

        return self.state_dict

    def reset_state(self):
        self.best_metrics = 0
        self.waitting = 0
        self.state_dict = {
            'save': False,
            'break': False
        }

