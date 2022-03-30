
# %%shell
# pip install transformers
# pip install sentencepiece
# wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bY-iaCn_CTaZE2-wp7_tJWPqocHLTC7y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bY-iaCn_CTaZE2-wp7_tJWPqocHLTC7y" -O BMDRecordGenerator.zip && rm -rf /tmp/cookies.txt
# unzip -o BMDRecordGenerator.zip

# French Record Generator
# ID: 1P0jl0BsU1d6gDJq2F1l4GNI-q9VYrIfK
# BMDGenerator(LIMIT: int, OUT_DIR: str, NOISE: float, TEMPLATES: list, DEBUG=False)

import BMDRecordGenerator

bmd = BMDRecordGenerator.BMDGenerator(400, "out", 5, ["marriage","birth"])
bmd.generate()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.nn.parameter import Parameter
import gc
import copy
from random import random as rand
from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel
import random

tokenizer = BertTokenizer.from_pretrained('./seed/tokenizer')#'bert-base-multilingual-cased')

def add_relation(relations, rev, search, entity, rtype):
  if search not in rev.keys():
    return

  for i in rev[search]:
    relations.append((entity[0], rtype, i))

ignore_label = set(['OtherDate', 'OtherName', 'SelfName', 'SpousePrefix'])
to_add_label = set()

def parse_relations(entities):
  #print(entities)

  relations = []

  rev = {}
  for e in entities:
    if e[1] in rev.keys():
      rev[e[1]].append(e[0])
    else:
      rev[e[1]] = [e[0]]

  for e in entities:
    if e[1] == 'FatherName':
      add_relation(relations, rev, 'SelfName', e, 'FatherOf')
      add_relation(relations, rev, 'MotherName', e, 'SpouseOf')
    elif e[1] == 'FatherAge':
      add_relation(relations, rev, 'FatherName', e, 'AgeOf')
    elif e[1] == 'SelfGender':
      add_relation(relations, rev, 'SelfName', e, 'GenderOf')
    elif e[1] == 'SpouseGender':
      add_relation(relations, rev, 'SpouseName', e, 'GenderOf')
    elif e[1] == 'BirthDate':
      add_relation(relations, rev, 'SelfName', e, 'BirthOf')
    elif e[1] == 'SpouseBirthDate':
      add_relation(relations, rev, 'SpouseName', e, 'BirthOf')
    elif e[1] == 'MotherName':
      add_relation(relations, rev, 'SelfName', e, 'MotherOf')
      add_relation(relations, rev, 'FatherName', e, 'SpouseOf')
    elif e[1] == 'MotherAge':
      add_relation(relations, rev, 'MotherName', e, 'AgeOf')
    elif e[1] == 'SpouseName':
      add_relation(relations, rev, 'SelfName', e, 'SpouseOf')
    elif e[1] == 'SelfName':
      add_relation(relations, rev, 'SpouseName', e, 'SpouseOf')
    elif e[1] == 'SpouseAge':
      add_relation(relations, rev, 'SpouseName', e, 'AgeOf')
    elif e[1] == 'SpouseMotherName':
      add_relation(relations, rev, 'SpouseName', e, 'MotherOf')
      add_relation(relations, rev, 'SpouseFatherName', e, 'SpouseOf')
    elif e[1] == 'SpouseMotherAge':
      add_relation(relations, rev, 'SpouseMotherName', e, 'AgeOf')
    elif e[1] == 'SpouseFatherAge':
      add_relation(relations, rev, 'SpouseFatherName', e, 'AgeOf')
    elif e[1] == 'SpouseFatherName':
      add_relation(relations, rev, 'SpouseName', e, 'FatherOf')
      add_relation(relations, rev, 'SpouseMotherName', e, 'SpouseOf')
    elif e[1] == 'MarriageDate':
      add_relation(relations, rev, 'SelfName', e, 'MarriageOf')
      add_relation(relations, rev, 'SpouseName', e, 'MarriageOf')
    elif e[1] == 'SelfAge':
      add_relation(relations, rev, 'SelfName', e, 'AgeOf')
    elif e[1] not in ignore_label:
      to_add_label.add(e[1])

  return relations

def parse_file(filename):
  file = open(filename, 'r')

  cur_tokens = []
  cur_entities = []
  simp_entities = []

  dataset = []

  state = 0
  simp_state = 0
  for line in file:
    parts = line.strip().split('\t')
    if len(parts) < 2:
      continue
    if parts[0] == 'None':
      dataset.append((cur_tokens, simp_entities, cur_entities, parse_relations(cur_entities)))
      state = 0
      cur_tokens = []
      cur_entities = []
      simp_entities = []

    #print(parts, cur_entities)

    parts[1] = parts[1].replace('Surname', 'Name')
    parts[1] = parts[1].replace('GivenName', 'Name')
    parts[1] = parts[1].replace('Person', 'Name')
    parts[1] = parts[1].replace('Year', 'Date')
    parts[1] = parts[1].replace('Month', 'Date')
    parts[1] = parts[1].replace('Day', 'Date')
    if parts[1] == 'Name':
      parts[1] = 'SelfName'

    if 'Name' in parts[1]:
      sub = 'Name'
    elif 'Date' in parts[1]:
      sub = 'Date'
    elif 'Gender' in parts[1]:
      sub = 'Gender'
    elif 'Age' in parts[1]:
      sub = 'Age'
    else:
      sub = 'None'
    #elif 'Prefix' in parts[1]:
    #  sub = 'Prefix'

    if parts[1] != 'none':
      if state != parts[1]:
        #cur_tokens.append('START=' + sub)
        state = parts[1]
        cur_entities.append((len(cur_tokens), state)) # Add start marker?
    else:
      state = 0

    if sub == 'None':
      bio = 0
    elif sub == simp_state:
      bio = 1
    else:
      bio = 2
    simp_state = sub

    tokens = tokenizer.tokenize(parts[0])

    cur_tokens.extend(tokens)

    for _ in range(len(tokens)):
      simp_entities.append((sub, bio))
      if bio == 2:
        bio = 1

  file.close()

  return dataset

aug = parse_file('out/frenchner.txt')
test = parse_file('data/all.tsv')

# relation_classes = set()
# for record in aug:
#   relation_classes.update([x[1] for x in record[2]])
# relation_classes = list(relation_classes)
# relation_classes.insert(0, 'None')
relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
rc2ind = {k: v for v, k in enumerate(relation_classes)}

ner_classes = ['None', 'Name', 'Date', 'Gender', 'Age']
ner2ind = {k: v for v, k in enumerate(ner_classes)}

bio_classes = ['Out', 'In', 'Beginning']

padded_length = 512

class REDataset(Dataset):
  def __init__(self, dataset):
    data = []
    run = 0
    for tokens, _, entities, relations in dataset:
      if len(tokens) > padded_length:
        #print('INPUT TOO LONG', len(tokens))
        continue
      if len(tokens) < padded_length:
        tokens.extend([''] * (padded_length - len(tokens)))
      ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

      checker = {}
      for a, b, c in relations:
        b = rc2ind[b]
        checker[(a,c)] = b

      for i, _ in entities:
        for j, _ in entities:
          if i==j:
            continue
          #if not (i, j) in checker:
          #  continue
          targ = torch.tensor(checker.get((i, j), 0))
          data.append((ids, torch.tensor(i), torch.tensor(j), targ))
      
    self.data = data
  
  def shuffle(self):
    random.shuffle(self.data)

  def __getitem__(self,i):
    return self.data[i]

  def __len__(self):
    return len(self.data)

  def transfer(self, other, amount):
    amount = min(amount, len(self.data))
    other.data.extend(self.data[-amount:])
    del self.data[-amount:]

  def subsample(self):
    groups = [[] for _ in range(len(relation_classes))]
    for it in self.data:
      groups[it[3].item()].append(it)
    maxsize = sorted([len(g) for g in groups])[-2] * 3
    self.data = []
    for g in groups:
      if len(g) > maxsize:
        random.shuffle(g)
        del g[maxsize:]
      self.data.extend(g)

class NERDataset(Dataset):
  def __init__(self, dataset):
    data = []
    run = 0
    for tokens, entities, _, _ in dataset:
      if len(tokens) > padded_length:
        #print('INPUT TOO LONG', len(tokens))
        continue
      if len(tokens) < padded_length:
        tokens.extend([''] * (padded_length - len(tokens)))
      ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)) # precompute this for sharing?

      for i, (e, b) in enumerate(entities):
        data.append((ids, torch.tensor(i), torch.tensor(ner2ind[e]), torch.tensor(b)))
      
    self.data = data
  
  def shuffle(self):
    random.shuffle(self.data)

  def __getitem__(self,i):
    return self.data[i]

  def __len__(self):
    return len(self.data)

  def transfer(self, other, amount):
    amount = min(amount, len(self.data))
    other.data.extend(self.data[-amount:])
    del self.data[-amount:]

  def subsample(self):
    groups = [[] for _ in range(len(ner_classes))]
    for it in self.data:
      groups[it[2].item()].append(it)
    maxsize = sorted([len(g) for g in groups])[-2] * 3
    self.data = []
    for g in groups:
      if len(g) > maxsize:
        random.shuffle(g)
        del g[maxsize:]
      self.data.extend(g)

#aug_dataset = REDataset(aug[:10])
#aug_dataset.subsample()
#test_dataset = NERDataset(aug[:5])
#test_dataset.subsample()

train_re_dataset = REDataset(aug)
val_re_dataset = REDataset(test)
test_re_dataset = REDataset([])

val_re_dataset.shuffle()
val_re_dataset.transfer(train_re_dataset, len(val_re_dataset) * 6 // 10)
val_re_dataset.transfer(test_re_dataset, len(val_re_dataset) * 1 // 2)

train_re_dataset.subsample()
val_re_dataset.subsample()
test_re_dataset.subsample()

train_ner_dataset = NERDataset(aug)
val_ner_dataset = NERDataset(test)
test_ner_dataset = NERDataset([])

val_ner_dataset.shuffle()
val_ner_dataset.transfer(train_ner_dataset, len(val_ner_dataset) * 6 // 10)
val_ner_dataset.transfer(test_ner_dataset, len(val_ner_dataset) * 1 // 2)

train_ner_dataset.subsample()
val_ner_dataset.subsample()
test_ner_dataset.subsample()

train_re_dataset_loader = DataLoader(train_re_dataset, batch_size=5, pin_memory=True, shuffle=True)
val_re_dataset_loader = DataLoader(val_re_dataset, batch_size=5, pin_memory=True, shuffle=True)
test_re_dataset_loader = DataLoader(test_re_dataset, batch_size=5, pin_memory=True, shuffle=True)

train_ner_dataset_loader = DataLoader(train_ner_dataset, batch_size=5, pin_memory=True, shuffle=True)
val_ner_dataset_loader = DataLoader(val_ner_dataset, batch_size=5, pin_memory=True, shuffle=True)
test_ner_dataset_loader = DataLoader(test_ner_dataset, batch_size=5, pin_memory=True, shuffle=True)

class JointModel(nn.Module):
  def __init__(self):
    super(JointModel, self).__init__()
    self.bert = BertModel.from_pretrained("./seed/bert", output_hidden_states=True) #"bert-base-multilingual-cased", output_hidden_states=True)
    size = 768 #9984 #768
    dropout = 0.5
    self.norm = nn.Sequential(
        nn.Dropout(dropout),
        nn.LayerNorm(size)
    )
    self.classify_re_left = nn.Linear(size, len(relation_classes))
    self.classify_re_right = nn.Linear(size, len(relation_classes))
    self.lstm = nn.LSTM(size, size, num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
    self.classify_ner = nn.Linear(size, len(ner_classes))
    self.classify_bio = nn.Linear(size, 3)

  def train_re(self, tokens, i, j):
    out = self.bert(tokens)
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)
    s = torch.arange(i.size()[0])
    subs = self.classify_re_left(out[s, i])
    objs = self.classify_re_right(out[s, j])
    return torch.add(subs, objs)

  def train_ner(self, tokens, i):
    out = self.bert(tokens)
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)
    #out = self.lstm(out)[0]
    s = torch.arange(i.size()[0])
    #print(out.shape)
    out = out[s, i]
    return self.classify_ner(out), self.classify_bio(out)

conf_re = np.zeros((len(relation_classes), len(relation_classes)), dtype=int)

conf_ner = np.zeros((len(ner_classes), len(ner_classes)), dtype=int)

conf_bio = np.zeros((3, 3), dtype=int)

def train_ner(model, objective, tokens, i, e, b, train):
  tokens, i, e, b = tokens.cuda(), i.cuda(), e.cuda(), b.cuda()

  ner_out, bio_out = model.train_ner(tokens, i)

  ner_preds = ner_out.argmax(1)
  bio_preds = bio_out.argmax(1)

  if train:
    loss = (objective(ner_out, e) +  objective(bio_out, b)) / 2
    loss.backward()
  else:
    #print(ner_preds, e)
    for yp, yt in zip(ner_preds.cpu().numpy(), e.cpu().numpy()):
      conf_ner[yt, yp] += 1
    for yp, yt in zip(bio_preds.cpu().numpy(), b.cpu().numpy()):
      conf_bio[yt, yp] += 1

  return (ner_preds == e).float().mean().item(), (bio_preds == b).float().mean().item()

def train_re(model, objective, tokens, i, j, truth, train):
  tokens, i, j, truth = tokens.cuda(), i.cuda(), j.cuda(), truth.cuda()
  
  y_hat = model.train_re(tokens, i, j)

  preds = y_hat.argmax(1)

  if train:
    loss = objective(y_hat, truth)
    loss.backward()
  else:
    for yp, yt in zip(preds.cpu().numpy(), truth.cpu().numpy()):
      conf_re[yt, yp] += 1

  return (preds == truth).float().mean().item()

def test(model, text, re_data_loader, ner_data_loader):
  global conf_re, conf_ner, conf_bio
  with torch.no_grad():
    conf_re *= 0
    accuracy_re = 0
    conf_ner *= 0
    accuracy_ner = 0
    conf_bio *= 0
    accuracy_bio = 0

    loop = tqdm(total=len(re_data_loader))
    batch = 0
    for re_data in re_data_loader:
      batch += 1

      accuracy_re += train_re(model, None, *re_data, False)

      loop.set_description('{}: re acc:{:.3f}'
        .format(text, accuracy_re / batch)) # replace epoch
      loop.update(1)

      #if batch > 1:
      #  break
    loop.close()
    accuracy_re /= batch

    loop = tqdm(total=len(ner_data_loader))
    batch = 0
    for ner_data in ner_data_loader:
      batch += 1
    
      sub_ner, sub_bio = train_ner(model, None, *ner_data, False)
      accuracy_ner += sub_ner
      accuracy_bio += sub_bio

      loop.set_description('{}: ner acc:{:.3f} bio acc:{:.3f}'
        .format(text, accuracy_ner / batch, accuracy_bio / batch))
      loop.update(1)

      #if batch > 1:
      #  break
    loop.close()
    accuracy_ner /= batch
    accuracy_bio /= batch

    print()
    print('Entities')
    print(conf_ner)
    print('Precisions')
    for i in range(len(ner_classes)):
      print(ner_classes[i], conf_ner[i, i] / sum(conf_ner[:,i]))
    print('Recalls')
    for i in range(len(ner_classes)):
      print(ner_classes[i], conf_ner[i, i] / sum(conf_ner[i]))

    print()
    print('Bios')
    print(conf_bio)
    print('Precisions')
    for i in range(3):
      print(bio_classes[i], conf_bio[i, i] / sum(conf_bio[:,i]))
    print('Recalls')
    for i in range(3):
      print(bio_classes[i], conf_bio[i, i] / sum(conf_bio[i]))

    print()
    print('Relationships')
    print(conf_re)
    print('Precisions')
    for i in range(len(relation_classes)):
      print(relation_classes[i], conf_re[i, i] / sum(conf_re[:,i]))
    print('Recalls')
    for i in range(len(relation_classes)):
      print(relation_classes[i], conf_re[i, i] / sum(conf_re[i]))

    return (2*accuracy_bio + accuracy_ner + accuracy_re) / 4

def scope():
  gc.collect()

  model = JointModel().cuda()

  #class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]).cuda()
  objective = nn.CrossEntropyLoss() #weight=class_weights) # try with this later?
  optimizer = optim.Adam([
                {'params': model.bert.parameters()},
                {'params': model.classify_re_left.parameters(), 'lr': 3e-4},
                {'params': model.classify_re_right.parameters(), 'lr': 3e-4},
                {'params': model.lstm.parameters(), 'lr': 3e-4},
                {'params': model.classify_ner.parameters(), 'lr': 3e-4},
                {'params': model.classify_bio.parameters(), 'lr': 3e-4}
              ], lr=3e-5)
  
  accuracy_re = 0

  accuracy_ner = 0

  accuracy_bio = 0

  epoch = 0
  lastbest = 0
  maxcombo = 0
  while epoch < 100:#for epoch in range(1,5):
    epoch += 1

    optimizer.zero_grad()
    #print(conf)

    accum = 5
    earlystop = accum * 300 # more efficient this way
    loop = tqdm(total=min([len(train_re_dataset_loader), len(train_ner_dataset_loader), earlystop]))
    batch = 0
    for re_data, ner_data in zip(train_re_dataset_loader, train_ner_dataset_loader):
      batch += 1

      accuracy_re += train_re(model, objective, *re_data, True)

      sub_ner, sub_bio = train_ner(model, objective, *ner_data, True)
      accuracy_ner += sub_ner
      accuracy_bio += sub_bio

      if batch % accum == 0:
        loop.set_description('training epoch {}: re acc:{:.3f} ner acc:{:.3f} bio acc:{:.3f}'.format(
            epoch, accuracy_re / accum, accuracy_ner / accum, accuracy_bio / accum))
        accuracy_re = 0
        accuracy_ner = 0
        accuracy_bio = 0
        optimizer.step()
        optimizer.zero_grad()

      loop.update(1)

      if batch == earlystop:
          break

    loop.close()

    combo = test(model, 'validation epoch ' + str(epoch), val_re_dataset_loader, val_ner_dataset_loader)

    print(combo)

    if combo > maxcombo:
      print('Saving model...')
      torch.save(model.state_dict(), 'bmodel')
      maxcombo = combo
      lastbest = 0
    else:
      lastbest += 1
      if lastbest >= 3:
        break

  # Done training
  print('Loading best model...')
  model.load_state_dict(torch.load('bmodel'))
  model.eval()

  print('Running test set')

  test(model, 'test set', test_re_dataset_loader, test_ner_dataset_loader)

scope()