import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.utils.rnn as rnn

START_TAG = '<start>'
END_TAG = '<end>'
PAD_TAG = '<pad>'


# BiLSTM+CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab, tag_vocab, pad,embedding_dim=256, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.pad = pad
        self.output_dim = len(self.tag_vocab)
        self.vocab_dim = len(self.word_vocab)
        self.transmission = nn.Parameter(torch.randn((self.output_dim, self.output_dim)))
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.embeddings = nn.Embedding(self.vocab_dim, embedding_dim)
        self.linear = nn.Linear(hidden_dim*2, self.output_dim)
        self.start_tag_index = self.tag_vocab[START_TAG]
        self.end_tag_index = self.tag_vocab[END_TAG]
        self.dropout = nn.Dropout(dropout)
        self.device = 'cuda'
        
        # never transfer to START tag and never transfer from END tag
        self.transmission.data[:, self.start_tag_index] = -10000000
        self.transmission.data[self.end_tag_index, :] = -10000000
    
    def forward(self, inputs, lengths):
        """
        计算inputs的feats
        :param inputs: [b_s, seq_len]
        :param lengths: [b_s]
        
        :return: [b_s, seq_len, out_dim]
        """
        embeddings = self.embeddings(inputs)
        inputs = rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        feats, (hn, cn) = self.LSTM(inputs)
        feats, lens_unpacked = rnn.pad_packed_sequence(feats, batch_first=True)
        feats_ = self.linear(feats)
        output = self.dropout(feats_)
        
        return output
    
    def score_sentence(self, feats, tags, masks):
        """
        给定目标语句的feats和标签，计算目标分数
        :param feats: [b_s, seq_len, out_dim]
        :param tags: [b_s, seq_len]
        :param masks: [b_s, seq_len]
        
        :return: [b_s]
        """
        forward = torch.gather(feats, 2, tags.unsqueeze(dim=2)).squeeze(dim=2).to(self.device) #[b_s, seq_len]
        forward[:, 1:] += self.transmission[tags[:, :-1], tags[:, 1:]]
        total_score = (forward*masks).sum(dim=1)
        return total_score
    
    def _forward_alg(self, feats, masks):
        """
        给定目标语句的feats和标签，使用viterbi算法，计算全部分数
        :param feats: [b_s,  seq_len, out_dim]
        :param masks: [b_s, seq_len]

        :return: [b_s]
        """
        # 初始情况start处的score为0
        score = torch.full((feats.shape[0], self.output_dim), -100000000).to(self.device)
        score[:, self.start_tag_index] = feats[:, 0, self.start_tag_index]
        # score = feats[:, 0]
        score = score.unsqueeze(dim=1) #[b_s, 1, out_dim]

        for i in range(1, feats.shape[1]):
            count = masks[:, i].sum(dim=0)
            count = count.type(torch.int)
            mid = feats[:count] #[count, seq_len, out_dim]
            feat = mid[:, i].unsqueeze(dim=2) #[count, out_dim, 1]
            sco = score[:count] + feat + self.transmission.T #[count, out_dim, out_dim]
            sco = torch.logsumexp(sco, dim=2).unsqueeze(dim=1)
            score = torch.cat((sco, score[count:]),dim=0) #[b_s, 1, out_dim]

        score = score.squeeze(dim=1) #[b_s, out_dim]
        return torch.logsumexp(score, dim=1) #[b_s]


    def viterbi(self, feats, masks):
        """
        输入特征向量，进行解码
        :param feats: [b_s, seq_len, out_dim]
        :masks: [b_s, seq_len]
        
        :return: (score, tag)
        """
        batch_size = feats.shape[0]
        forward = [[[i] for i in range(self.output_dim)]] * batch_size #[b_s, out_dim, 1]
        d = feats[:, 0].unsqueeze(dim=1) #[b_s, 1, out_dim]
        for i in range(1, feats.shape[1]):
            n_unfinished = masks[:, i].sum()
            n_unfinished = n_unfinished.type(torch.int)
            score = feats[:n_unfinished, i].unsqueeze(dim=2) + self.transmission.T #[n_un, out_dim, out_dim]
            score += d[:n_unfinished] #[n_un, out_dim, out_dim]
            score_max ,index_max = torch.max(score, dim=2) #[n_un, out_dim]
            index_max = index_max.tolist() 
            forward[:n_unfinished] = [[forward[b_s][k]+[j] for j,k in enumerate(p)] for b_s, p in enumerate(index_max)] #[n_un, out_dim, ~]
            d = torch.cat((score_max.unsqueeze(dim=1), d[n_unfinished:]), dim=0) #[b_s, 1, out_dim]
        d = d.squeeze()
        score_max, index_max = torch.max(d, dim=1) #[b_s]
        tags = [forward[i][j] for i,j in enumerate(index_max)]
        return score_max, tags

    def neg_log_likelihood(self, sentences, tags, lengths):
        """
        给定句子和标签，计算损失
        :param sentences: [b_s, seq_len]
        :param tags: [b_s, seq_len]
        :param lengths: [b_s]

        :return: [b_s]
        """
        feats = self.forward(sentences, lengths)
        mask = (sentences != self.pad).to(self.device).type(torch.float)
        prob_score = self.score_sentence(feats, tags, mask)
        gold_score = self._forward_alg(feats,mask)
        # print(prob_score - gold_score)
        return gold_score - prob_score

    def _predict(self, sentences, lengths):
        """
        给定句子和长度，计算最大概率的标签
        :param sentences: [b_s, seq_len]
        :param lengths: [b_s]

        :return (score, tags): [b_s]
        """ 
        feats = self.forward(sentences, lengths)
        mask = (sentences!=self.pad).to(self.device).type(torch.float)
        score_max, tags = self.viterbi(feats, mask)

        return score_max, tags