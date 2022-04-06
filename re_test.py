
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
import gc
import copy
from random import random as rand
from transformers import BertTokenizer, BertModel
import random

from joint.model import JointModel

import joint.dataset # todo: clean this up with relation_classes use

relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
rc2ind = {k: v for v, k in enumerate(relation_classes)}

ner_classes = ['None', 'Name', 'Date', 'Gender', 'Age']
ner2ind = {k: v for v, k in enumerate(ner_classes)}

bio_classes = ['Out', 'In', 'Beginning']

train_re_dataset_loader, val_re_dataset_loader, test_re_dataset_loader, train_ner_dataset_loader, val_ner_dataset_loader, test_ner_dataset_loader = joint.dataset.load_datasets()

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

  model = JointModel(relation_classes, ner_classes).cuda()

  #class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]).cuda()
  objective = nn.CrossEntropyLoss() #weight=class_weights) # try with this later?
  optimizer = optim.Adam([
                {'params': model.bert.parameters()},
                #{'params': model.classify_re_left.parameters(), 'lr': 3e-4},
                #{'params': model.classify_re_right.parameters(), 'lr': 3e-4},
                #{'params': model.classify_re.parameters(), 'lr': 3e-4},
                #{'params': model.lstm.parameters(), 'lr': 3e-4},
                {'params': model.classify_ner.parameters(), 'lr': 3e-4},
                {'params': model.classify_bio.parameters(), 'lr': 3e-4}
              ], lr=3e-5)
  
  accuracy_re = 0

  accuracy_ner = 0

  accuracy_bio = 0
  accum = 5
  earlystop = accum * 300 # more efficient this way

  epoch = 0
  lastbest = 0
  maxcombo = 0
  while epoch < 100:#for epoch in range(1,5):
    epoch += 1

    optimizer.zero_grad()
    #print(conf)

    loop = tqdm(total=min([len(train_ner_dataset_loader), earlystop]))
    batch = 0
    for ner_data in train_ner_dataset_loader:
      batch += 1

      sub_ner, sub_bio = train_ner(model, objective, *ner_data, True)
      accuracy_ner += sub_ner
      accuracy_bio += sub_bio

      if batch % accum == 0:
        loop.set_description('training ner epoch {}: ner acc:{:.3f} bio acc:{:.3f}'.format(
            epoch, accuracy_ner / accum, accuracy_bio / accum))
        accuracy_ner = 0
        accuracy_bio = 0
        optimizer.step()
        optimizer.zero_grad()

      loop.update(1)

      if batch == earlystop:
          break

    loop.close()

    combo = test(model, 'validation ner epoch ' + str(epoch), val_re_dataset_loader, val_ner_dataset_loader)

    print(combo)

    if combo > maxcombo:
      print('Saving model...')
      torch.save(model.state_dict(), 'bmodel')
      maxcombo = combo
      lastbest = 0
    else:
      lastbest += 1
      if lastbest >= 5:
        break

  print('Loading best model...')
  model.load_state_dict(torch.load('bmodel'))
  model.bert2 = copy.deepcopy(model.bert) # weight transfer

    #class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1]).cuda()
  objective = nn.CrossEntropyLoss() #weight=class_weights) # try with this later?
  optimizer = optim.Adam([
                {'params': model.bert2.parameters()},
                {'params': model.classify_re_left.parameters(), 'lr': 3e-4},
                {'params': model.classify_re_right.parameters(), 'lr': 3e-4},
                {'params': model.classify_re.parameters(), 'lr': 3e-4},
                #{'params': model.lstm.parameters(), 'lr': 3e-4},
                #{'params': model.classify_ner.parameters(), 'lr': 3e-4},
                #{'params': model.classify_bio.parameters(), 'lr': 3e-4}
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
    
    loop = tqdm(total=min([len(train_re_dataset_loader), earlystop]))
    batch = 0
    for re_data in train_re_dataset_loader:
      batch += 1

      accuracy_re += train_re(model, objective, *re_data, True)

      if batch % accum == 0:
        loop.set_description('training re epoch {}: re acc:{:.3f}'.format(
            epoch, accuracy_re / accum, accuracy_ner / accum, accuracy_bio / accum))
        accuracy_re = 0
        optimizer.step()
        optimizer.zero_grad()

      loop.update(1)

      if batch == earlystop:
          break

    loop.close()

    combo = test(model, 'validation re epoch ' + str(epoch), val_re_dataset_loader, val_ner_dataset_loader)

    print(combo)

    if combo > maxcombo:
      print('Saving model...')
      torch.save(model.state_dict(), 'bmodel')
      maxcombo = combo
      lastbest = 0
    else:
      lastbest += 1
      if lastbest >= 5:
        break

  # Done training
  print('Loading best model...')
  model.load_state_dict(torch.load('bmodel'))
  model.eval()

  print('Running test set')

  test(model, 'test set', test_re_dataset_loader, test_ner_dataset_loader)

scope()