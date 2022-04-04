import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import copy

class JointModel(nn.Module):
  def __init__(self, re_classes, ner_classes):
    super(JointModel, self).__init__()
    self.re_classes = re_classes
    self.ner_classes = ner_classes

    self.bert = BertModel.from_pretrained("./seed/bert", output_hidden_states=True) #"bert-base-multilingual-cased", output_hidden_states=True)
    self.bert2 = copy.deepcopy(self.bert)
    size = 768 #9984 #768
    dropout = 0.5
    self.norm = nn.Sequential(
        nn.Dropout(dropout),
        nn.LayerNorm(size)
    )
    #self.classify_re_left = nn.Linear(size, len(re_classes))
    #self.classify_re_right = nn.Linear(size, len(re_classes))
    self.classify_re_left = nn.Linear(size, size)
    self.classify_re_right = nn.Linear(size, size)
    self.classify_re = nn.Sequential(
      nn.Linear(size, len(re_classes))
    )
    #self.lstm = nn.LSTM(size, size, num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
    self.classify_ner = nn.Linear(size, len(ner_classes))
    self.classify_bio = nn.Linear(size, 3)

  # TODO: issues with multiple of same relation
  def train_re(self, tokens, i, j):
    out = self.bert2(tokens)
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)
    s = torch.arange(i.size()[0])

    subs = self.classify_re_left(out[s, i])
    objs = self.classify_re_right(out[s, j])
    out = subs * objs # normalize this?
    return self.classify_re(out)

  def train_ner(self, tokens, i):
    out = self.bert(tokens)
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)
    #out = self.lstm(out)[0]
    s = torch.arange(i.size()[0])
    #print(out.shape)
    out = out[s, i]
    return self.classify_ner(out), self.classify_bio(out)

  def compute(self, tokens):
    out = self.bert(tokens.unsqueeze(0))
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)

    out = out[0] # extract sentence  
  
    ner = F.softmax(self.classify_ner(out), dim=1)
    bio = F.softmax(self.classify_bio(out), dim=1)

    words, num_ner = ner.shape

    # TODO: beam search here instead?
    state = 0
    start, end = 0, 0
    entities = []
    for i in range(words):
      if state > 0:
        if ner[i][state] > 0.5: # continue
          end += 1
        else: # end
          entities.append([start, end, state])
          state = 0
      if i-start > 1: # block single tokens as unlikely
        for j in range(1, num_ner):
          if state != j and (ner[i][j] > 0.5):
            if state != 0:
              if end == i:
                end = i-1
              entities.append([start, end, state])
            #print(state, j, len(entities), ner[i].numpy(), bio[i].numpy())
            state = j
            start = i
            end = i
            break
    # append last

    out = self.bert2(tokens.unsqueeze(0))
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)[0]

    left = self.classify_re_left(out)
    right = self.classify_re_right(out)

    relations = []
    for i in range(len(entities)):
      le = entities[i]
      for j in range(len(entities)):
        if i == j:
          continue
        re = entities[j]
        sub = self.classify_re(left[le[0]] * right[re[0]]) # todo: normalize this?
        #sub = self.classify_re(torch.cat([out[le[0]], out[re[0]]], dim=1))
        #sub = left[le[0]] + right[re[0]]
        soft = F.softmax(sub, dim=0)

        # TODO: add masking constraints and recalculate
        

        if soft[0] < 0.8: # check this
          soft[0] = 0
        argmax = torch.argmax(soft).item()
        #print(i, j, argmax, soft[argmax])
        if argmax > 0:
          #print(i, j, argmax, soft[argmax])
          relations.append((i, argmax, j))

    # Order events, people, people attributes
    # Will break if order of relations changes
    #order = {s: i for i, s in enumerate(['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf'])}
    #relations = sorted(relations, key=lambda x: order[self.re_classes(x[1])])
    relations = sorted(relations, key=lambda x: (-x[1], -abs(x[0] - x[2])))

    #print(relations)

    for l, c, r in relations:
      if c == 7: # Marriage
        entities[l][2] = 'MarriageDate'
        entities[r][2] = 'SelfName'
      elif c == 6: # Birth
        entities[l][2] = 'BirthDate'
        entities[r][2] = 'SelfName'
      elif c == 5: # Spouse
        # l -> r due to symmetry so that first spouse is selfname
        if type(entities[l][2]) == int or entities[l][2] == 'SelfName': # don't process parents
          entities[l][2] = 'SelfName'
          entities[r][2] = 'SpouseName'
      elif c == 4: # Mother
        if type(entities[r][2]) == int or entities[r][2] == 'SelfName':
          entities[r][2] = 'SelfName'
          entities[l][2] = 'MotherName'
        elif entities[r][2] == 'SpouseName':
          entities[l][2] = 'SpouseMotherName'
      elif c == 3: # Father
        if type(entities[r][2]) == int or entities[r][2] == 'SelfName':
          entities[r][2] = 'SelfName'
          entities[l][2] = 'FatherName'
        elif entities[r][2] == 'SpouseName':
          entities[l][2] = 'SpouseFatherName'
      elif c == 2: # Age
        if type(entities[r][2]) == str and entities[r][2][-4:] == 'Name':
          entities[l][2] = entities[r][2][:-4] + 'Age'
      elif c == 1: # Gender
        if type(entities[r][2]) == str and entities[r][2][-4:] == 'Name':
          entities[l][2] = entities[r][2][:-4] + 'Gender'

    for e in entities:
      if type(e[2]) == int:
        e[2] = 'Other' + self.ner_classes[e[2]]

    # Todo: name and date splitting
    #print('\n'.join(map(str, relations)))

    return entities
