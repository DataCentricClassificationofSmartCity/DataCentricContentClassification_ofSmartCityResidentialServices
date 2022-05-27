import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
import pickle
from word_match.dataset.sequence_label_dataloader import SeqLabelDataloader
from word_match.transformer.seq_label_transformer import SeqLableTransformer

test_path = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/test_set.csv'
train_path = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/train_set.csv'
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)

# Load data loader
dataloader = SeqLabelDataloader(train_data, test_data, batch_size=128, eval_portion=0.1, cut_length=100)

# get word2vec embedding list, add 2 for <pad> and <eos>
vocab_size = np.load('../../data/eos_token.pickle', allow_pickle=True) +2

save_name = 'NLP_Transformer'

# Implement model
if not os.path.exists('models/' + save_name):
    os.makedirs('models/' + save_name)
if not os.path.exists('logs/' + save_name):
    os.makedirs('logs/' + save_name)

model = SeqLableTransformer('models/' + save_name, 'logs/' + save_name, n_layers=3, n_head=2,
                            d_features=50, d_meta=None, enc_seq_length=100, dec_seq_length=12,
                            d_k=100, d_v=100, d_bottleneck=128, n_vocab=vocab_size)

# Training
model.train(200, dataloader.train_dataloader(), dataloader.val_dataloader(), device='cpu', save_mode='best',
            smoothing=True, earlystop=True)
