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
  - Remove punctuation, stopwords, etc by [LTP](https://github.com/HIT-SCIR/ltp).
- **Label Proprocessing**
  - Label Analysis
  - Label Statistic
- **Word Embedding and Vectorization**
  - Word embedding by Word2Vec
  - Word embedding by Fasttext
  - Word embedding by Bert
  - Word embedding by XLNet
 
Due to the privacy, the dataset is provided in a tokenization format. Which is after word embedding and vectorization processing.

# Classification Models

The classifiction Models in this work are ResNet, Transformer and XLNet. In Transformer, we choose two strategries named based transformer(3 Encoder-Decoder Layers) and enlarged-transformer (6 Encoder-Decoder Layers with a cross-layer sharing parameter group).

- **[ResNet](https://github.com/KaimingHe/deep-residual-networks)**
- **[Transformer](https://github.com/huggingface/transformers)**
  - Based Transformer
  - Enlarged Transformer
- **[XLNet](https://github.com/zihangdai/xlnet)**

# Word Matching

Word matching is a workflow that covert the request and department natural language to tokens and place as a pair (X,Y) to train the transformer model. And The model will learn to generate a sentence in a token format aiming at the department. This sentence is then to fuzzy matching with the department in the dataset. Comparied for the classification, the training pair (X,Y) is request and the classes label index.

# Attention Value Calcluction

This work aims to observe the different attention value words' influence on the decision of the transformer based model. After calculating the attention value of each words, low attention words are masked to do the prediction of the model. 

# Examples on Colabs.

Classification, Word Matching and Attention Value calcuction tasks are available on Colab.
