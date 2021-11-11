import torch
import torch.nn as nn
from torch import optim

START_TAG = "<START>"
END_TAG = "<END>"
embedding_dims = 5
hidden_dims = 4


# 计算路径总概率
def log_sum_exp(score):
    max_score = torch.max(score)
    return max_score+torch.log(torch.sum(torch.exp(score - max_score)))


class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, token_to_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = len(token_to_idx)
        self.token_to_idx = token_to_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.trans_matrix = nn.Parameter(torch.randn((self.output_dim, self.output_dim)))
        self.linear = nn.Linear(hidden_dim * 2, self.output_dim)
        # 这两句约束不会转移到 START_TAG 和 END_TAG
        # self.trans_matrix.data[:, self.token_to_idx[START_TAG]] = -1000000
        # self.trans_matrix.data[:, self.token_to_idx[END_TAG]] = -1000000

    def forward_alg(self, feats):
        # 初始时刻START_TAG 处的概率为1
        forward_score = torch.full((self.output_dim,), -1000000).to(device)
        forward_score[self.token_to_idx[START_TAG]] = 0

        for feat in feats:
            temp_score = []
            for i in range(self.output_dim):
                # 转移分数, 每个状态转移到i状态的分数
                transfer = self.trans_matrix[:, i]
                # 发射分数, 由当前词向量预测出i状态的分数
                emit = feat[i]
                # 计算出每条路径上的分数
                mid_score = forward_score + transfer + emit
                prob = log_sum_exp(mid_score)
                temp_score.append(prob)
            forward_score = torch.stack(temp_score)

        forward_score += self.trans_matrix[:, self.token_to_idx[END_TAG]]
        return log_sum_exp(forward_score)

    def compute_score(self, feats, tags):
        total_score = 0
        former_tag = self.token_to_idx[START_TAG]
        for i in range(len(tags)):
            total_score += self.trans_matrix[former_tag, tags[i]] + feats[i, tags[i]]
            former_tag = tags[i]
        total_score += self.trans_matrix[tags[len(tags)-1], self.token_to_idx[END_TAG]]
        return total_score

    def forward(self, sentence):
        embedding = self.embedding(sentence)
        embedding = embedding.unsqueeze(0)
        hidden, (hn, cn) = self.lstm(embedding)
        feats = self.linear(hidden)
        return feats.squeeze()

    def compute_loss(self, sentence, tags):
        feats = self.forward(sentence)
        score = self.forward_alg(feats)
        target_score = self.compute_score(feats, tags)
        return score - target_score

    def viterbi(self, sentence):
        feats = self.forward(sentence)
        back_list = []
        forward_score = torch.full((self.output_dim,), -100000).to(device)
        forward_score[self.token_to_idx[START_TAG]] = 0
        for feat in feats:
            back_ = []
            temp_score = []
            for i in range(self.output_dim):
                score = forward_score + self.trans_matrix[:, i] + feat[i]
                max_id = torch.argmax(score)
                back_.append(max_id)
                temp_score.append(score[max_id])
            forward_score = torch.stack(temp_score)
            back_list.append(back_)

        # 反向遍历
        forward_score += self.trans_matrix[:, self.token_to_idx[END_TAG]]
        max_id = torch.argmax(forward_score)
        max_score = torch.max(forward_score)
        return_list = [max_id]
        back_list.reverse()
        for item in back_list:
            return_list.append(item[max_id])
            max_id = item[max_id]
        return_list.reverse()
        keys = list(self.token_to_idx.keys())
        tokens = [keys[i] for i in return_list]
        return tokens, max_score


token = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}
training_data = [("the wall street journal reported today that apple corporation made money".split(),
                  "B I I I O O O B I O O".split()), ("georgia tech is a university in georgia".split(),
                                                     "B I O O O O B".split())]

word_dict = {}
pd = 0
for item in training_data:
    for word in item[0]:
        if word not in word_dict.keys():
            word_dict[word] = pd
            pd = pd+1
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model(len(word_dict), embedding_dims, hidden_dims, token).to(device)
# print("初次测试：{}".format(model.viterbi(torch.tensor([word_dict[i] for i in training_data[0][0]]).to(device))))
optimizer = optim.Adam(model.parameters())

for i in range(300):
    for pair in training_data:
        sentence = torch.tensor([word_dict[j] for j in pair[0]]).to(device)
        target = torch.tensor([token[j] for j in pair[1]]).to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 计算 loss
        loss = model.compute_loss(sentence, target)
        # 梯度反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()


# 预测
with torch.no_grad():
    print("再次测试：{}".format(model.viterbi(torch.tensor([word_dict[i] for i in training_data[0][0]]).to(device))))
    print("再次测试：{}".format(model.viterbi(torch.tensor([word_dict[i] for i in training_data[1][0]]).to(device))))







