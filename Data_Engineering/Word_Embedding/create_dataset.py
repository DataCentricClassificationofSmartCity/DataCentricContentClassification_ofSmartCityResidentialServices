from Data_Preprocessing.word_embeddings.utils import write_lines
import pandas as pd

dataset_0 = pd.read_csv('/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/train_22_Mar_2021.csv')['sentence_no_symbol']
dataset_1 = pd.read_csv('/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/test_22_Mar_2021.csv')['sentence_no_symbol']
dataset_2 = pd.read_csv('/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/val_22_Mar_2021.csv')['sentence_no_symbol']

dataset = pd.concat([dataset_0, dataset_1, dataset_2])
del dataset_0
del dataset_1
del dataset_2

output_file = '../../data/word2vec_training_data_Mar25.txt'

write_lines(dataset, output_file, batch_size=100)