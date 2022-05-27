from word_match.dataset.utils import *
import pandas as pd
pd.options.display.max_rows = 999
from sklearn.model_selection import train_test_split
import re
import pickle

vocab = {'<ukw>':0}
# Load data
label_data = pd.read_csv('../data/label_comparision.csv')
raw_data = pd.read_csv('../data/8910_split_loc_dpt.csv', encoding='gb18030')
raw_data = raw_data.dropna(subset=['第一级名称.1'])
raw_data['raw_label'] = raw_data['第一级名称.1']
# Create index sequence dataset
df, vocab = create_index_sequence(raw_data, vocab)
print(len(vocab))
df, vocab = create_index_label(df, raw_data, vocab)
print(len(vocab))

eos_token = len(vocab)
df = add_eos_token(df, eos_token)
df.to_csv('../data/index_sequence_dataset.csv')

# Cleaning the dataset, drop empty or short sequences
df = pd.read_csv('../data/index_sequence_dataset.csv')
# df = keep_length(df, (1, 100))

# Split datasets and save
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv('../data/train_set.csv')
test.to_csv('../data/test_set.csv')

with open('../data/vocab.pickle', 'wb') as ev:
    pickle.dump(vocab, ev)

with open('../data/eos_token.pickle', 'wb') as et:
    pickle.dump(eos_token, et)