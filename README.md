# Data Centric Content Classification of Smart City Residential Services

Implementation of the publication Data Centric Content Classification of Smart City Residential Services.

# Environment Requirment

The following versions are requried to perform the tasks.

- python>=3.7.0
- tensorflow-gpu>=2.1.0, <3.0.0
- keras>=2.3.1, <3.0.0
- pandas>=0.25.3, <2.0.0
- scikit-learn>=0.22.1, <2.0.0
- gensim>=3.8.1, <4.0.0

Training and testing run on Google Colab Pro Service. 

# Introduction

This implementation contains whole tasks that demonstrated in the publication. The main component are:

- **Data Engineering**
- **Classification Models**
- **Word Matching**
- **Attention Value Calcluction**
- **Examples on Colab**

# Data Engineering

Data engineering processes and converts the data set from Chinese texts to word vectors as the input of machine learning models. The label analyzing and statistic is also included in this component.

- **Data Proprocessing**
  - Segmente into tokens
  - Remove punctuation, stopwords, etc by [LTP]().
- **Label Proprocessing**
  - Label Analysis
  - Label Statistic
- ** Word Embedding and Vectorization**
  - Word embedding by Word2Vec
  - Word embedding by Fasttext
  - Word embedding by Bert
  - Word embedding by XLNet
 
Due to the privacy, the dataset is provided in a tokenization format. Which is after word embedding and vectorization processing.

# Classification Models

The classifiction Models in this work are ResNet, Transformer and XLNet. In Transformer, we choose two strategries named base-transformer(3 Encoder-Decoder Layers) and enlarged-transformer (6 Encoder-Decoder Layers with a cross-layer sharing parameter group).

- **[ResNet](https://github.com/KaimingHe/deep-residual-networks)**
- **[Transformer](https://github.com/huggingface/transformers)**
- **[XLNet](https://github.com/zihangdai/xlnet)**
