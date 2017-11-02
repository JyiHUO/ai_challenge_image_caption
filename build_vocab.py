# -*-coding=utf-8-*-

"""
目的:构建词典
步骤:
- 去掉低频词
- 为每一个词赋值id
返回：
向文件写vacab类(pickle形式)
"""

import pickle
import argparse
from collections import Counter
import jieba
import json

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx +=1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_path, threshold):
    """
    实现所有功能
    :param json: json文件路径
    :param threshold: 稀缺词的词频
    :return: vocab类
    """
    with open(json_path, 'r') as f:
        json_list = json.load(f)

    counter = Counter()
    for i, item in enumerate(json_list):
        """
        item:(image_id, caption)
        """
        caption = item[1]
        counter.update(jieba.cut(caption))

        if i % 1000 == 0:
            print ("You have finish 1000 image caption, hurry up!!!")

    words = [word for word, cnt in counter.items() if cnt > threshold]

    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(json_path=args.caption_path,
                        threshold=args.threshold)

    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print ("total vovab size: %d"%(len(vocab)))
    print ("your vocab file had save to '%s'" %(vocab_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_path",type=str,
                        default="format_caption_train_annotations_20170902.json",
                        help="path for json file")
    parser.add_argument("--vocab_path", type=str,
                        default="vocab.pkl",
                        help="path for saving file")
    parser.add_argument("--threshold", type=int,
                        default=4,
                        help="minimum word count should discard")

    args = parser.parse_args()
    main(args)

