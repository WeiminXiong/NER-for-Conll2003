import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

def read_data(file_path):
    """
    读取数据
    :param file_path: 文件名
    
    :return: [(sentence, tag)...]
    """
    list = []
    sentence = ['<START>']
    tag = ['<START>']
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过标题
            if(line == "-DOCSTART- -X- -X- O"):
                pass
            line.strip('\n')
            # 遇到空行
            if(len(line) == 0):
                
