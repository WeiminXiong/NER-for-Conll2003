import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from tqdm.std import trange
from model import BiLSTM_CRF
from dataset import read_data, collate_fn, conll2003, vocab, split_data_and_tag, combine_data_and_tag
import os

Train = False

# 预定义
num_epoch = 50
batch_size = 64
embedding_dim = 256
hidden_dim = 256
drop_out = 0.5
lr = 1e-3

data_path = "./day_3/conll2003_v2"
# data_path = "./conll2003_v2"
train_data_path = os.path.join(data_path, "train.txt")
valid_data_path = os.path.join(data_path, "valid.txt")
test_data_path = os.path.join(data_path, 'test.txt')
save_path = './day_3/model/first.pth'

# 构造字典
before_manage_train_data = read_data(train_data_path)
before_manage_valid_data = read_data(valid_data_path)
before_manage_test_data = read_data(test_data_path)
# print(before_manage_train_data[0])
# print(len(before_manage_train_data))
train_sentence, train_tag = split_data_and_tag(before_manage_train_data)
valid_sentence, valid_tag = split_data_and_tag(before_manage_valid_data)
test_sentence, test_tag = split_data_and_tag(before_manage_test_data)
# print(train_sentence[0],train_sentence[1], train_tag[0], train_tag[1])
sentence_vocab = vocab.build(train_sentence)
tag_vocab = vocab.build(train_tag)

# 编码数据
train_data = combine_data_and_tag(train_sentence, train_tag, sentence_vocab, tag_vocab)
valid_data = combine_data_and_tag(valid_sentence, valid_tag, sentence_vocab, tag_vocab)
test_data = combine_data_and_tag(test_sentence, test_tag, sentence_vocab, tag_vocab)

# 训练
model = BiLSTM_CRF(sentence_vocab, tag_vocab, sentence_vocab.pad, embedding_dim, hidden_dim, drop_out).to('cuda')
optimizer = Adam(model.parameters(), lr)
train_dataset = conll2003(train_data)
valid_dataset = conll2003(valid_data)
test_dataset = conll2003(test_data)
# train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn = collate_fn(sentence_vocab.pad, tag_vocab.pad))
# valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, collate_fn = collate_fn(sentence_vocab.pad, tag_vocab.pad))

# sentence, tag, length = next(iter(train_dataloader))
# # print(sentence[0], tag[0], length[0])
# print(model.neg_log_likelihood(sentence.to('cuda'), tag.to('cuda'), length))

epoch_bar = tqdm(desc="training_routine", total=num_epoch, position=0)
train_bar = tqdm(desc="train", total=len(train_data)//batch_size +1, position=1, leave=True)
valid_bar = tqdm(desc="valid", total=len(valid_data)//batch_size +1, position=1, leave=True)
model.load_state_dict(torch.load(save_path))

if Train:
    for epoch in range(num_epoch):
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn = collate_fn(sentence_vocab.pad, tag_vocab.pad))
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, collate_fn = collate_fn(sentence_vocab.pad, tag_vocab.pad))
        train_loss_sum = 0
        train_batch_size = 0
        valid_loss_sum = 0
        valid_batch_size = 0
        running_loss = 0
        
        # 训练
        model.train()
        for sentences, tags, lengths in train_dataloader:
            sentences=sentences.to('cuda')
            tags = tags.to('cuda')
            optimizer.zero_grad()
            batch_loss = model.neg_log_likelihood(sentences, tags, lengths)
            train_loss_sum += batch_loss.sum().item()
            train_batch_size += len(sentences)
            loss = batch_loss.mean()
            loss.backward()
            optimizer.step()
            running_loss = train_loss_sum / train_batch_size

            train_bar.set_postfix(loss=running_loss, epoch=epoch)
            train_bar.update()

        # 验证
        running_loss = 0
        model.eval()
        with torch.no_grad():
            for sentences, tags, lengths in valid_dataloader:
                sentences = sentences.to('cuda')
                tags = tags.to('cuda')
                batch_loss = model.neg_log_likelihood(sentences, tags, lengths)
                valid_loss_sum += batch_loss.sum().item()
                valid_batch_size += len(sentences)
                
                running_loss = valid_loss_sum/ valid_batch_size
                valid_bar.set_postfix(loss=running_loss, epoch=epoch)
                valid_bar.update()
        
        train_bar.n = 0
        valid_bar.n = 0
        epoch_bar.update()
        torch.save(model.state_dict(), save_path)
else:
    model.eval()
    test_bar = tqdm(desc='test', total= len(test_data)//batch_size+1)
    test_dataloader = DataLoader(test_dataset, batch_size, False, collate_fn = collate_fn(sentence_vocab.pad, tag_vocab.pad))
    result_file_path = 'result.txt'
    with open(result_file_path, 'w') as result_file:
        for sentences, tags, lengths in test_dataloader:
            sentences = sentences.to('cuda')
            tags = tags.to('cuda')
            score, predict_tags = model._predict(sentences, lengths)
            for sentence, tag, predict_tag in zip(sentences, tags, predict_tags):
                sentence, tag, predict_tag = sentence[1: -1], tag[1: -1], predict_tag[1:-1]
                for token, true_tag, predict in zip(sentence, tag, predict_tag):
                    result_file.write(' '.join([sentence_vocab.index_to_token[token], tag_vocab.index_to_token[true_tag], tag_vocab.index_to_token[predict]])+ '\n')
                result_file.write('\n')
            test_bar.update()