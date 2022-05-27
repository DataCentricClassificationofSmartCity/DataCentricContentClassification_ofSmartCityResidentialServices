from fuzzywuzzy import process
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
def match_class(w2v_model, pred_list, dict):
    '''
    Match words to class
    :param w2v_model: word2vec model (gensim)
    :param pred_list: words prediction list
    :param dict: label dictionary
    :return:
    '''
    pred_label = []
    pred_sentence = []
    for word_idxes in tqdm(pred_list):
        sentence = []
        # Index to word
        for word_idx in word_idxes:
            sentence.append(w2v_model.wv.index2word[word_idx])
        # Join words and form a sentence
        label = ''.join(sentence)
        # find best match
        best_match = process.extractOne(label, dict['label_list'])[0]
        # get class id
        idx = dict['label_dict'][best_match]
        pred_label.append(idx)
        pred_sentence.append(label)
    return pred_label, pred_sentence

def match_preset_labels(w2v_model, pred_list):
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

    all_labels_pure = [x.replace(',', '') for x in all_labels]
    dict = {}
    for label in all_labels:
        dict[label.replace(',', '')] = label

    encoder = LabelEncoder()
    encoder.fit(all_labels)
    pred_label = []
    for word_idxes in pred_list:
        sentence = []
        # Index to word
        for word_idx in word_idxes:
            sentence.append(w2v_model.wv.index2word[word_idx])
        # Join words and form a sentence
        label = ''.join(sentence)
        best_match = process.extractOne(label, all_labels_pure)[0]

        origin_label = dict[best_match]
        id = encoder.transform([origin_label])[0]
        pred_label.append(id)

    return pred_label
