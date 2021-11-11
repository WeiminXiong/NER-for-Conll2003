import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

START_TAG = '<START>'
END_TAG = '<END>'
PAD_TAG = '<PAD>'


# BiLSTM+CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab, tag_vocab, embedding_dim=256, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.output_dim = len(self.tag_vocab)
        self.vocab_dim = len(self.word_vocab)
        self.transmission = nn.Parameter(torch.randn((self.output_dim, self.output_dim)))
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.embeddings = nn.Embedding(self.vocab_dim, embedding_dim)
        self.linear = nn.Linear(hidden_dim*2, self.output_dim)
        self.start_tag_index = self.tag_vocab[START_TAG]
        self.end_tag_index = self.tag_vocab[END_TAG]
        self.dropout = nn.Dropout(dropout)
        
        # never transfer to START tag and never transfer from END tag
        self.transmission[:, self.start_tag_index] = -10000000
        self.transmission[self.end_tag_index, :] = -10000000
    
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
        forward = torch.gather(feats, 2, tags.unsqueeze(dim=2)).squeeze(dim=2) #[b_s, seq_len]
        forward[:, 1:] += self.transmission[tags[:, :-1], tags[:, 1:]]
        total_score = (forward*masks.type(torch.float)).sum(dim=1)
        return total_score
    
    def _forward_alg(self, feats, masks):
        """
        给定目标语句的feats和标签，计算全部分数
        :
        """