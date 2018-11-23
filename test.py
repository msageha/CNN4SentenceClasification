# coding:utf-8
import janome
from janome.tokenizer import Tokenizer
import MeCab

import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import FastText

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from model import EncoderRNN, AttnClassifier
import random
random.seed(0)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

tagger = MeCab.Tagger('')
tagger.parse('')

# j_t = Tokenizer()
# def tokenizer(text): 
#     return [tok for tok in j_t.tokenize(text, wakati=True)]

def tokenizer(text):
    wakati = []
    node = tagger.parseToNode(text).next
    while node.next:
        wakati.append(node.surface)
        node = node.next
    return wakati

#Fieldクラス
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True)
LABEL = data.LabelField(sequential=False)
FILE = data.LabelField(sequential=False)

#データの読み込み
dataset = data.TabularDataset(path='./document.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL), ('File', FILE)], skip_header=True)

LABEL.build_vocab(dataset)
FILE.build_vocab(dataset)

train, val, test = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())

TEXT.build_vocab(train, vectors=FastText(language="ja"), min_freq=2)

#size
print(TEXT.vocab.vectors.size())

# device = torch.device('cpu')
device = torch.device('cuda:0')
train_iter, val_iter, test_iter = data.Iterator.splits((train, val, test), batch_sizes=(16, 16, 1), device=device, repeat=False,sort=False)

batch = next(iter(train_iter))
print(batch.Text)
print(batch.Label)

def test_model(epoch, test_iter):
    encoder.eval()
    classifier.eval()
    correct = 0
    for idx, batch in enumerate(test_iter):
        (x, x_l), y = batch.Text, batch.Label
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
    print('test epoch:{}, acc:{}'.format(epoch, correct/float(len(test))))

gpu = True
emb_dim = 300 #単語埋め込み次元
h_dim = 3 #lstmの隠れ層の次元
class_num = len(LABEL.vocab.itos) #予測クラス数
lr = 0.001 #学習係数
epochs = 50 #エポック数

 # make model
encoder = EncoderRNN(emb_dim, h_dim, len(TEXT.vocab), gpu=gpu, v_vec = TEXT.vocab.vectors)
classifier = AttnClassifier(h_dim, class_num)

if gpu:
    encoder = encoder.cuda()
    classifier = classifier.cuda()

# init model
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

for m in encoder.modules():
    print(m.__class__.__name__)
    weights_init(m)
    
for m in classifier.modules():
    print(m.__class__.__name__)
    weights_init(m)

def save_model(epoch):
    print('___save_model___')
    torch.save(encoder.state_dict(), f'./model/encoder_{epoch}.pkl')
    torch.save(classifier.state_dict(), f'./model/classifer_{epoch}.pkl')

def load_model(epoch):
    print('___load_model___')
    encoder.load_state_dict(torch.load(f'./model/encoder_{epoch}.pkl'))
    classifier.load_state_dict(torch.load(f'./model/classifer_{epoch}.pkl'))

# optim
from itertools import chain
optimizer = optim.Adam(chain(encoder.parameters(),classifier.parameters()), lr=lr)

load_epoch = 3
load_model(load_epoch)

# test model
test_model(load_epoch, test_iter)
