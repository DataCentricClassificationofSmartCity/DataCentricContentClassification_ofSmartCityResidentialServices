import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from ltp import LTP

pd.options.display.max_rows = 999

tqdm.pandas()
should_drop = ['市民来电', '请职能局按规定在30分钟内', '核实处理', '联系市民', '响应处置']

ltp = LTP()


# def cut_sentence(sentence):
#     sentence = re.sub("[\s+\.\!\/\-_,$%^*()+\"\']+|[a-zA-Z0-9+——！，。？、~@#￥%……&*（）《》：:]+", "", sentence)
#     for i in should_drop:
#         sentence = re.sub(i, '', sentence)
#     words = jieba.cut_for_search(sentence)
#     word_list = []
#     for word in words:
#         word_list.append(word)
#
#     return word_list


# def sentence_to_word_indexes(sentence, vocab):
#     '''
#         Take a sentence, cut it into words and convert them to indexes.
#
#         Arguments:
#             sentence {String} -- a sentence
#             vocab {dictionary} -- word-embedding dictionary
#
#         Returns:
#             indexes {String} -- a list of word indexes in string format
#     '''
#     indexes = []
#     words = cut_sentence(sentence)
#     for word in words:
#         try:
#             index = vocab[word]
#         except:
#             vocab[word] = len(vocab)
#             index = vocab[word]
#         indexes.append(index)
#
#     return str(indexes)


def add_bos_eos(df, bos_token, eos_token):
    def add_token(x):
        x.insert(0, bos_token)
        x.insert(len(x), eos_token)
        return x

    df['target_indexes'] = df['target_indexes'].progress_apply(add_token)
    return df


def process_labels(targets):
    '''
        Convert label to numerical indicators

        Arguments:
            targets {Pandas Series} -- labels in string format

        Returns:
            classes {Pandas Series} -- converted label indicators
    '''
    encoder = LabelEncoder()
    classes = targets.value_counts().index
    encoder.fit(classes)
    classes = pd.Series(encoder.transform(targets), name='class_index')
    return classes


def sentence_to_indexes(raw_data, words, vocab, ukn):
    '''
        Create data set

        Arguments:
            raw_data {Pandas Dataframe} -- input Dataframe contains column '诉求内容' and '处置单位（处理后）'
            vocab {Dictionary} -- word embedding dictionary

        Returns:
            new_data {Pandas Dataframe} -- 4 columns, input word sequence, target word sequence, target class and the origin class
    '''

    x_data = pd.DataFrame(columns=['input_indexes', 'target_indexes', 'class', 'class_origin'])
    # x_data['target_indexes'] = raw_data['output_words'].apply(to_indexes, args=(vocab, True))
    x_data['target_indexes'] = raw_data['origin_output'].progress_apply(to_indexes, args=(vocab, True, ukn))
    x_data['input_indexes'] = words['words'].progress_apply(to_indexes, args=(vocab, False, ukn))
    x_data['class'] = process_labels(raw_data['output_words'])
    x_data['class_origin'] = raw_data['origin_output']
    return x_data, vocab


def get_vocab(Model, path):
    '''
        Get vocab from a Gensim model

        Arguments:
            Model {Class} -- gensim model
            path {String} -- model path

        Returns:
            vocab {Dictionary}
    '''
    model = Model.load(path)
    vocab = model.wv.vocab
    return vocab


def keep_length(data, range=(0, 999)):
    '''
        Keep those input sequence with length in `range`

        Arguments:
            data {Pandas DataFrame}
            range {tuple} -- sequence length range

        Returns:
            data {Pandas DataFrame}
    '''
    print('Before: %d' % data.shape[0])
    data['sequence_length'] = data['input_indexes'].map(lambda x: len(eval(x)))
    # print(data['sequence_length'].value_counts().sort_index())
    data = data[data['sequence_length'] >= range[0]]
    data = data[data['sequence_length'] <= range[1]]
    print('After: %d' % data.shape[0])
    return data


def to_indexes(words, vocab, output_label=True, ukn=-1):
    indexes = []
    if output_label:
        words, _ = ltp.seg([words])
        words = words[0]
    else:
        words = words.split(' ')
    for word in words:
        try:
            indexes.append(vocab[word].index)
        except:
            indexes.append(ukn)
    return indexes


def process_raw_data(raw_data, words, vocab):
    ukn_token = len(vocab)
    bos_token = len(vocab) + 1
    eos_token = len(vocab) + 2
    df, vocab = sentence_to_indexes(raw_data, words, vocab, ukn_token)
    df = add_bos_eos(df, bos_token, eos_token)
    return df
