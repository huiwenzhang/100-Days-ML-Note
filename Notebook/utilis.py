"""
Inlucde code related to how to read data from text and generate batch training
data from dataset
"""

import numpy as np
import copy
import pickle


def batch_generator(arr, batch_size, n_steps):
    """
    Generate a batch of training samples
    :param arr: shape(none, 1), the whole  training data, each int number represents a char
    :param batch_size: batch size
    :param n_steps: sequence length
    :return: x: shape(batch_size, num_steps)
    :return: y: shape(batch_size, num_steps)
    """
    arr = copy.copy(arr)
    # flat sequence steps
    total_batch_samples = batch_size * n_steps
    n_batchs = int(len(arr) / total_batch_samples)
    # truncate to make sure the len of arr is proper
    arr = arr[:total_batch_samples * n_batchs]
    # to make sure the row is equal to batch_size
    arr = arr.reshape((batch_size, -1))
    while True:
        np.random.shuffle(arr)  # shuffle batch sequence
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]  # target is one-step lag behind the input
            yield x, y


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, vocab_file=None):
        """

        :param text: training text object
        :param max_vocab: maxmimun number of letters
        """
        # we don't save text as an attribute of the class because the text is
        # used for extract chars, if we load chars from a given file, then there
        #     is no need to read all the text into memory any more
        if vocab_file is not None:
            with open(vocab_file, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print('Number of vocab', len(vocab))
            vocab_cnt = dict.fromkeys(list(vocab), 0)
            for char in text:
                vocab_cnt[char] += 1
            # we don't rank the original  vocab_cnt because it is a unordered dict
            vocab_cnt_list = []
            for char in vocab_cnt:
                vocab_cnt_list.append((char, vocab_cnt[char]))
            vocab_cnt_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_cnt_list) > max_vocab:
                vocab_cnt_list = vocab_cnt_list[:max_vocab]
            self.vocab = [x[0] for x in vocab_cnt_list]

        self.char2int_table = {c: i for i, c in
                               enumerate(self.vocab)}  # convert char to index for one-hot use
        self.int2char_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        # +1 stands for abnormal chars
        return len(self.vocab) + 1

    def char2int(self, char):
        if char in self.char2int_table:
            return self.char2int_table[char]
        else:
            return len(self.vocab)

    def int2char(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int2char_table[index]
        else:
            raise Exception('Unknown index')

    def text2arr(self, text):
        """
        Convert text to arr, text is char, arr includes corresponding index
        Because the text is not an attribute, so one argument is needed
        :param text:
        :return:
        """
        arr = []
        for char in text:
            arr.append(self.char2int(char))
        return np.array(arr)

    def arr2text(self, arr):
        text = []
        for index in arr:
            text.append(self.int2char(index))
        return "".join(text)

    def save_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
