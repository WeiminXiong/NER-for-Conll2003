from numpy.lib.index_tricks import index_exp
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

def read_data(file_path):
    """
    读取数据
    :param file_path: 文件名
    
    :return: [(sentence, tag)...]
    """
    list = []
    sentence = ['<start>']
    tag = ['<start>']
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过标题
            if line == '-DOCSTART- -X- -X- O\n':
                continue
            line = line.strip('\n')
            # 遇到空行
            if len(line) == 0:
                if len(sentence)>1:
                    sentence.append('<end>')
                    tag.append('<end>')
                    list.append((sentence, tag))
                    sentence =['<start>']
                    tag = ['<start>']
                else:
                    pass
            else:
                li = line.split(' ')
                sentence.append(li[0])
                tag.append(li[-1])
    return list

class conll2003(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class collate_fn:
    """
    预处理batch中的数据
    """
    def __init__(self, sen_pad, tag_pad):
        self.sen_pad = sen_pad
        self.tag_pad = tag_pad
    
    def __call__(self, ex):
        """
        :param ex: [b_s]

        :return (sentences, tags, lengths): [b_s]
        """
        length = np.array([len(e[0]) for e in ex])
        sorted_length_index = length.argsort()[::-1].tolist()
        sorted_batch = [ex[i] for i in sorted_length_index]
        sentences =pad_sequence([torch.tensor(item[0]) for item in sorted_batch], batch_first=True, padding_value=self.sen_pad)
        tags = pad_sequence([torch.tensor(item[1]) for item in sorted_batch], batch_first=True, padding_value=self.tag_pad)
        lengths = length[sorted_length_index]
        return sentences, tags, lengths


class vocab:
    """
    构造词表映射
    :attr token_to_index: 词语到索引的映射
    :attr index_to_token: 索引到词语的映射
    :attr pad: <pad> 的标签
    :attr unk: <unk> 的标签
    """
    def __init__(self, tokens = None):
        self.token_to_index = dict()
        self.index_to_token = list()

        if tokens!=None:
            if '<unk>' not in tokens:
                tokens = ['<unk>'] + tokens
            if '<pad>' not in tokens:
                tokens = ['<pad>'] + tokens
            for item in tokens:
                self.token_to_index[item] = len(self.index_to_token)
                self.index_to_token.append(item)    
            self.pad = self.token_to_index['<pad>']
            self.unk = self.token_to_index['<unk>']
    
    @classmethod
    def build(cls, sentences, min_freq = 1, reserved_tokens = None):
        """
        生成需要索引的词语
        """
        token_count = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                token_count[word] += 1
        unique_token = ['<unk>', '<pad>'] + (reserved_tokens if reserved_tokens else [])
        unique_token += [token for token, freq in token_count.items() if freq >= min_freq and token!='<unk>' and token!='<pad>']
        return cls(unique_token)

    def __len__(self):
        return len(self.index_to_token)
    
    def  __getitem__(self, token):
        return  self.token_to_index.get(token, self.unk)
    
    def convert_tokens_to_indexes(self, tokens):
        """
        生成句子对应的索引
        """
        return [self[token] for token in tokens]
    
    def convert_indexes_to_tokens(self, indexes):
        """
        生成索引对应的句子
        """
        return [self.index_to_token[index] for index in indexes]        

def split_data_and_tag(li):
    """
    将list data拆分为句子和tag
    :return: sentences, tags
    """
    sentences = []
    tags = []
    for item in li:
        sentences.append(item[0])
        tags.append(item[1])
    return sentences, tags

def combine_data_and_tag(sentences, tags, s_vocab, t_vocab):
    """
    将训练数据和标签绑定并编码
    :param sentences: 句子列表
    :param tags: 标签列表
    :param s_vocab: 句子词典
    :param t_vocab: 标签词典
    :return: list[(sentence, tag)]
    """
    li = []
    length = len(sentences)
    for i in range(length):
        sentence = [s_vocab[item] for item in sentences[i]]
        tag = [t_vocab[item] for item in tags[i]]
        li.append((sentence, tag))
    return li