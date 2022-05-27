import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score

from src.training.transformer_dataloader import TextualDataloader
from src.training.transformer_model import TransformerClassifierModel
from src.training.transformer_utils import get_embeddings

# Load data loader
dataloader = TextualDataloader('/content/drive/My Drive/NLP_gov/history_data/train_set.csv',
                               '/content/drive/My Drive/NLP_gov/history_data/eval_and_test.csv',
                               batch_size=128, eval_portion=0.2, cut_length=100)

# get word2vec embedding list
embedding, vector_length = get_embeddings('/content/drive/My Drive/NLP_gov/history_data/history_word_embedding.CBOW',
                                          padding=True)

save_name = 'NLP_Transformer_'

# Implement model
if not os.path.exists('models/' + save_name):
    os.makedirs('models/' + save_name)
if not os.path.exists('logs/' + save_name):
    os.makedirs('logs/' + save_name)

model = TransformerClassifierModel('models/' + save_name, 'logs/' + save_name, embedding=embedding,
                                   max_length=dataloader.max_length, n_classes=dataloader.n_targets,
                                   d_features=100, d_k=100, d_v=100, d_meta=None, n_layers=3, n_head=8, dropout=0.1,
                                   d_classifier=256, d_bottleneck=128)

# Training
model.train(200, dataloader.train_dataloader(), dataloader.val_dataloader(), device='cuda', save_mode='best',
            smoothing=True, earlystop=True)

# model.load_model('/content/drive/My Drive/NLP_gov/history_data/nlp-encoder-100--step-25900_loss-0.49195')

predictions = model.get_predictions(data_loader=dataloader.test_dataloader(), device='cuda')
accuracy_score(predictions[1], np.array(predictions[0]).argmax(axis=1))
precision_score(predictions[1], np.array(predictions[0]).argmax(axis=1), average='weighted')

micro_recall = recall_score(predictions[1], np.array(predictions[0]).argmax(axis=1), average='micro')
macro_recall = recall_score(predictions[1], np.array(predictions[0]).argmax(axis=1), average='macro')
micro_prec = precision_score(predictions[1], np.array(predictions[0]).argmax(axis=1), average='micro')
macro_prec = precision_score(predictions[1], np.array(predictions[0]).argmax(axis=1), average='macro')
print('micro_recall:', micro_recall, '\nmacro_recall:', macro_recall,
      '\nmicro_prec:', micro_prec, '\nmacro_prec:', macro_prec)
