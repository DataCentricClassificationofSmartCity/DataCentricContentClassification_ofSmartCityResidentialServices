import multiprocessing
from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence


def count_words(file_path):
    # return word list and count unique words
    words_list = []
    with open(file_path, 'r', encoding='gb18030') as f:
        for line in f:
            for word in line.split():
                words_list.append(word)

    return words_list, len(set(words_list))


def train_model(training_file, Model, save_path, vector_size=100, window=5, min_count=0):
    '''
        Train a Gensim word embedding model

        Arguments:
            training_file {String} -- training file path (produced by write_lines())
            Model {Class} -- gensim model
            save_path {String} -- model save path
    '''
    model = Model(LineSentence(training_file), size=vector_size, window=window, workers=multiprocessing.cpu_count(),
                  min_count=min_count)
    model.save(save_path)


class Embedding:
    def __init__(self, model_path, Model):
        self.model = Model.load(model_path)

    def __call__(self, words):
        return self.model[words]

#
# training_file = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/data/word2vec_data_Mar27.txt'
# save_path = '/Users/sunjincheng/Desktop/Programs/NLP_smart_dispatching/models/CBOW_Mar27.model'
# words_list, count = count_words(training_file)
# train_model(training_file, Word2Vec, save_path)
