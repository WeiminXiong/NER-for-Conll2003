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
            if line == "-DOCSTART- -X- -X- O":
                pass
            line.strip('\n')
            # 遇到空行
            if len(line) == 0:
                if len(sentence)>1:
                    list.append((sentence.append('<edn>'), tag.append('<end>')))
                    sentence =['<START>']
                    tag = ['<START>']
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
        sorted_batch = ex[sorted_length_index]
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