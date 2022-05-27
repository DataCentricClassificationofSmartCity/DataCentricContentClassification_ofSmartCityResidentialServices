import torch
import os
import pandas as pd
from gensim.models.word2vec import Word2Vec

from word_match.dataset.sequence_label_dataloader import SeqLabelDataloader
from word_match.transformer.seq_label_transformer import SeqLableTransformer

val_path = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/val_data_Apr6.csv'
test_path = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/test_data_Apr6.csv'
train_path = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/train_data_Apr6.csv'
val_data = pd.read_csv(val_path)[:200]
test_data = pd.read_csv(test_path)[:50]
train_data = pd.read_csv(train_path)[:200]

# Load data loader
dataloader = SeqLabelDataloader(train_data, None, val_data, batch_size=128, cut_length=100, out_len=13, in_len=100,
                                test=False)

vocab = Word2Vec.load('/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/CBOW_Mar29.model')

# get word2vec embedding list, add 2 for <pad> and <eos>
vocab_size = len(vocab.wv.vocab) + 4

save_name = 'NLP_Transformer'

# Implement model
if not os.path.exists('models/' + save_name):
    os.makedirs('models/' + save_name)
if not os.path.exists('logs/' + save_name):
    os.makedirs('logs/' + save_name)

model = SeqLableTransformer(save_name, save_name, n_layers=3, n_head=8,
                            d_features=100, d_meta=None, enc_seq_length=100, dec_seq_length=dataloader.output_length,
                            d_k=100, d_v=100, d_bottleneck=128, n_vocab=vocab_size, embedding=vocab.wv.vectors)

test_dataloader = SeqLabelDataloader(None, test_data, None, batch_size=128, cut_length=100, out_len=13, in_len=100,
                                     test=True)

# model.load_model('/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/models/model-step-59500-loss-19.35002', 'cpu')
pred_list, real_list, enc_slf_attn, dec_enc_attn = model.get_predictions(test_dataloader.test_dataloader(), 'cpu', 13,
                                                                         bos_token=50278, eos_token=50279,
                                                                         greedy=True)

test_attn = enc_slf_attn[0][0]


def cal_word_avg_attn(attn):
    n_head = 8
    attn_layer_3 = attn[2]
    attn_layer_2 = attn[1]
    attn_layer_1 = attn[0]
    attn_2_1 = torch.bmm(attn_layer_2.squeeze(0), attn_layer_1.squeeze(0))
    attn_3_21 = torch.bmm(attn_layer_3.squeeze(0), attn_2_1)
    attn_w2w = attn_3_21.sum(0) / n_head
    attn_word = attn_w2w.mean(0)
    return attn_word


attn_list = []

# Calculate average attention each word get
for attn in enc_slf_attn:
    attn_list.append(cal_word_avg_attn(attn[0]))
#
mask_list = []
# Example threshold
n_words = 4
threshold = 0.2 / n_words
for attn in attn_list:
    mask_list.append(attn.gt(threshold))





# enc_slf_attn
test_attn = enc_slf_attn[0][0]
import tensorflow as tf

def cal_word_avg_attn(attn):
    n_head = 8
    attn_layer_3 = attn[2]
    attn_layer_2 = attn[1]
    attn_layer_1 = attn[0]
    attn_2_1 = torch.bmm(attn_layer_2.squeeze(0), attn_layer_1.squeeze(0))
    attn_3_21 = torch.bmm(attn_layer_3.squeeze(0), attn_2_1)
    attn_w2w = attn_3_21.sum(0) / n_head
    attn_word = attn_w2w.mean(0)
    return attn_word

def cal_mask_list(enc_slf_attn,percent = 90):
  attn_list = []
  # Calculate average attention each word get
  for attn in enc_slf_attn:
      attn_list.append(cal_word_avg_attn(attn[0]))

  mask_list = []
  for attn in attn_list:
    threshold = np.percentile(attn.cpu().numpy(),percent)
    mask_list.append(attn.gt(threshold).reshape(100,1))
  return mask_list

mask_list = cal_mask_list(enc_slf_attn,90)
# np.shape(mask_list[0])
# np.shape(mask_list[0].reshape(1,100,1))
np.shape(mask_list[0])