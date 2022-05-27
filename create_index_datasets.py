import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Data_Preprocessing.create_dataset import sentence_to_word_indexes, get_vocab, keep_length


def process_labels(targets):
    encoder = LabelEncoder()
    classes = targets.value_counts().index
    print('labels are:', classes)
    encoder.fit(classes)
    classes = pd.Series(encoder.transform(targets), name='class_index')
    return classes


def create_index_sequence(input_data, vocab):
    new_data = pd.DataFrame(columns=['input_indexes', 'target_indexes', 'class', 'class_origin'])
    new_data['input_indexes'] = input_data['诉求内容'].apply(sentence_to_word_indexes, args=(vocab,))
    new_data['target_indexes'] = input_data['ltp_dpt'].apply(sentence_to_word_indexes, args=(vocab,))
    new_data['class'] = process_labels(input_data['ltp_dpt'])
    new_data['class_origin'] = input_data['ltp_dpt']
    return new_data


vocab = get_vocab(Word2Vec, 'history_word_embedding.CBOW')
raw_data = pd.read_csv('data/final_data.csv')

# Create index sequence datasets
new_data = create_index_sequence(raw_data, vocab)
new_data.to_csv('datasets/index_sequence_dataset.csv')

# Cleaning the datasets, drop empty or short sequences
new_data = keep_length(new_data, (1, 300))
new_data.reset_index(drop=True, inplace=True)

all_idx = new_data.index.to_list()
for i in range(14):
    left, current_train = train_test_split(new_data, test_size=44000)
    current_train, current_test = train_test_split(current_train, test_size=4000)
    current_train.reset_index(drop=True, inplace=True)
    current_test.reset_index(drop=True, inplace=True)
    print('current train set size:', len(current_train),
          '; current test set size:', len(current_test),
          '; left size:', len(left))
    current_train.to_csv('datasets/train_set_' + str(i) + '.csv')
    current_test.to_csv('datasets/test_set_' + str(i) + '.csv')
    new_data = left
