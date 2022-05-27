import csv
import re
from tqdm import tqdm

from ltp import LTP

ltp = LTP()

should_drop = ['市民来电', '请职能局按规定在30分钟内', '核实处理', '联系市民', '响应处置']


def cut_sentence(sentence):
    sentence = re.sub("[\s+\.\!\/\-_,$%^*()+\"\']+|[a-zA-Z0-9+——！，。？、~@#￥%……&*（）《》：:]+", "", sentence)
    for i in should_drop:
        sentence = re.sub(i, '', sentence)
    words, hidden = ltp.seg(sentence)
    word_list = []
    for word in words:
        word_list.append(word)
    if not word_list:
        return sentence
    else:
        return word_list


def cut_sentence_ltp(sentence):
    words, _ = ltp.seg(sentence.tolist())
    return words


def write_lines(input_file, output_file, batch_size=100, starting_batch=0):
    '''
        Read sentences from `input file` line by line, cut it into words and write them in `output_file`
        For training embedding models

        Arguments:
            input_file {DataFrame} -- input file
            output_file {String} -- output file path
    '''
    write_file = open(output_file, 'a', encoding='gb18030')
    n_batches = input_file.shape[0] // batch_size + 1
    for i in tqdm(range(starting_batch, n_batches)):
        if i == n_batches - 1:
            sentences = input_file[i * batch_size:]
        else:
            sentences = input_file[i * batch_size: (i + 1) * batch_size]
        word_lists = cut_sentence_ltp(sentences)
        for wl in word_lists:
            words = ' '.join(wl)
            write_file.write(words)
            write_file.write('\n')

    write_file.close()
