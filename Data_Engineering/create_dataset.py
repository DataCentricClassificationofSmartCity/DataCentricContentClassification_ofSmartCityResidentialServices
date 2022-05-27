from gensim.models import Word2Vec, FastText
import pandas as pd
import re
pd.options.display.max_rows = 999
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def words_to_list(sentence, vocab):
    '''
        Take a sentence, cut it into words and convert them to indexes.

        Arguments:
            sentence {String} -- a sentence
            vocab {dictionary} -- word-embedding dictionary

        Returns:
            indexes {String} -- a list of word indexes in string format
    '''
    indexes = []
    words = re.split(r'[,\s]', sentence)
    for word in words:
        try:
            index = vocab[word].index
        except:
            # <unknown word>
            index = -1
        indexes.append(index)
    return str(indexes)


def process_labels(targets):
    '''
        Convert label to numerical indicators

        Arguments:
            targets {Pandas Series} -- labels in string format

        Returns:
            classes {Pandas Series} -- converted label indicators
    '''
    all_labels = ['公安局,交通,警察,支队', '公安局', '城管局', '政府', '交通,运输,和,港航,管理局', '街道办', '住建局',
                  '工商局', '公共,交通,集团,有限公司', '教育局', '供电局', '港航,控股,有限公司', '社会保险,事业局',
                  '国土,资源局', '市政管理', '环保局', '其他', '卫生局', '劳动保障,监察,大队', '食品,药品,监督,管理局',
                  '国家,税务局', '人力,资源,和,社会,保障局', '物价,监督,检查局', '水务局', '电信局', '环卫局', '移动,分公司',
                  '民政局', '地方,税务局', '威立雅,水务,有限公司', '公安局,消防,支队', '经济,开发区', '国际,机场',
                  '开源,水务,资产,管理,有限公司', '排水,管道,养护所', '改造,项目,指挥部', '公共,绿化,管理所',
                  '民生,燃气,股份,有限公司', '质量,技术,监督,局', '物价局', '劳动,监察,大队', '海汽,运输,集团,股份,有限公司',
                  '规划,委员会', '旅游,文化,投资,控股,集团,有限公司', '园林,管理局', '联合,网络,通信,股份,有限公司,分公司',
                  '住房,保障,中心', '有线,电视,网络,有限公司,分公司', '城建,集团,有限公司', '国土,资源,执法,大队', '司法局',
                  '烟草,专卖局', '房屋,征收局', '商务局', '旅游,发展,委员会', '镇政府', '城建设,投资,有限公司',
                  '海洋,和,渔业局', '文体局', '人口,计划生育,委员会', '重点,项目,推进,管理,委员会', '京环,城环境,服务,有限公司',
                  '市民,云,支撑组', '园林局', '宽带,网络,服务,有限公司', '林业局', '港务,公安局',
                  '国家,高新技术,产业,开发区,管理,委员会', '地下,综合,管廊,投资,管理,有限公司', '文化,体育,和,旅游,发展局',
                  '房屋,交易,与,产权,管理,中心', '菜篮子,产业,集团,有限,责任,公司', '安全,生产,监督,管理局']
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    classes = pd.Series(encoder.transform(targets), name='class_index')
    return classes


def create_index_sequence(input, target, vocab):
    '''
        Create data set

        Arguments:
            input: Input sentence with " " as the word separation
            target: Target sentence
            vocab {Dictionary} -- word embedding dictionary

        Returns:
            new_data {Pandas Dataframe} -- 4 columns, input word sequence, target word sequence, target class and the origin class
    '''
    new_data = pd.DataFrame(columns=['input_indexes', 'target_indexes', 'class', 'class_origin'])
    new_data['input_indexes'] = input['words'].apply(words_to_list, args=(vocab,))
    new_data['target_indexes'] = target['output_words'].apply(words_to_list, args=(vocab,))
    new_data['class'] = process_labels(target['output_words'])
    new_data['class_origin'] = target['origin_output']
    return new_data


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


# # Load embedding vocab
# vocab = get_vocab(Word2Vec, '../models/word_embedding.CBOW')
#
# Load data
# label_data = pd.read_csv('../data/label_comparision.csv')
# raw_data = pd.read_csv('../data/8910_split_loc_dpt.csv', encoding='gb18030')
# raw_data['处置单位（处理后）'] = label_data['after_ltp_dpt']
#
# # Create index sequence dataset
# new_data = create_index_sequence(raw_data, vocab)
# new_data.to_csv('../data/index_sequence_dataset.csv')
#
# # Cleaning the dataset, drop empty or short sequences
# new_data = pd.read_csv('../data/index_sequence_dataset.csv')
# new_data = keep_length(new_data, (1, 300))
#
# # Split datasets and save
# train, test = train_test_split(new_data, test_size=0.2, random_state=42)
# train.to_csv('../data/train_set.csv')
# train.to_csv('../data/test_set.csv')


def generate_index_dataset(input_cutwords, raw_data, vocab):
    return create_index_sequence(input_cutwords, raw_data, vocab)


