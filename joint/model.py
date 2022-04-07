import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import copy

class JointModel(nn.Module):
  def __init__(self, re_classes, ner_classes):
    super(JointModel, self).__init__()
    self.re_classes = re_classes
    self.rc2ind = {k: v for v, k in enumerate(re_classes)}
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
    self.classify_ner = nn.Linear(size, len(ner_classes))
    self.classify_bio = nn.Linear(size, 3)
    self.tokenizer = None

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

  def load_tokenizer(self, path=None):
    if path is None:
      self.tokenizer = BertTokenizer.from_pretrained('./seed/tokenizer')

  def compute(self, record):

    if self.tokenizer is None:
      raise Exception('Load the tokenizer first')

    tokens_raw = self.tokenizer.tokenize(record)

    tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens_raw))

    out = self.bert(tokens.unsqueeze(0))
    out = out[2][-1] #torch.cat(out[2], dim=2) # test this on all hidden layers
    out = self.norm(out)

    out = out[0] # extract sentence  
  
    ner = F.softmax(self.classify_ner(out), dim=1)
    bio = F.softmax(self.classify_bio(out), dim=1)

    words, num_ner = ner.shape

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

        # show = torch.argmax(soft).item() > 0
        # if show:
        #   print(le,re, soft.numpy())
        #   print(self.re_classes[torch.argmax(soft).item()])
        #   print(i, self.ner_classes[le[2]], j, self.ner_classes[re[2]])

        if self.ner_classes[le[2]] != 'Gender' or self.ner_classes[re[2]] != 'Name':
          soft[self.rc2ind['GenderOf']] = 0
        if self.ner_classes[le[2]] != 'Age' or self.ner_classes[re[2]] != 'Name':
          soft[self.rc2ind['AgeOf']] = 0
        if self.ner_classes[le[2]] != 'Name' or self.ner_classes[re[2]] != 'Name':
          soft[self.rc2ind['FatherOf']] = 0
          soft[self.rc2ind['MotherOf']] = 0
          soft[self.rc2ind['SpouseOf']] = 0
        if self.ner_classes[le[2]] not in ['Year', 'Month', 'Day'] or self.ner_classes[re[2]] != 'Name':
          soft[self.rc2ind['BirthOf']] = 0
          soft[self.rc2ind['MarriageOf']] = 0

        # if show:
        #   print(soft.numpy())

        soft /= soft.sum()

        #if soft[0] < 0.8: # check this
        #  soft[0] = 0
        argmax = torch.argmax(soft).item()
        #print(i, j, argmax, soft[argmax])
        if argmax > 0:
          #print(i, j, argmax, soft[argmax])
          relations.append((i, argmax, j, soft[argmax].item()))

    # Order events, people, people attributes
    # Will break if order of relations changes
    # Parse relations in decreasing order of probability
    relations = sorted(relations, key=lambda x: (-x[1], -x[3]))

    #print(relations)

    block = set()
    for l, c, r, p in relations:
      print(l, c, r, p)
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

    def only_first(parts, s):
      wc = 0
      piece = ''
      while s > 0 and wc < 3:
        s -= 1
        if tokens_raw[s].startswith('##'):
          piece = tokens_raw[s][2:] + piece
        else:
          piece = tokens_raw[s] + piece
          if piece.startswith('prenom') or piece.startswith('prénom'):
            return True
          wc += 1
          piece = ''
      return len(parts) == 1

    answer = []
    print(tokens_raw)
    for s, e, i in entities:
        print(s, e, i)
        while tokens_raw[s].startswith('##'):
          s -= 1
        while e + 1 < len(tokens_raw) and tokens_raw[e+1].startswith('##'):
          e += 1

        text = ' '.join(tokens_raw[s:e+1]).replace(' ##', '')
        if i == 'OtherName':
          i = 'OtherPerson'

        if i.endswith('Name'):
          parts = text.split()
          ind = -1
          if only_first(parts, s):
            answer.append((' '.join(parts), i[:-4] + 'GivenName'))
          else:
            answer.append((' '.join(parts[:-1]), i[:-4] + 'GivenName'))
            print(answer[-1])
            answer.append((' '.join(parts[-1:]), i[:-4] + 'Surname'))
        else:
          answer.append((text, i))

        print(answer[-1])

    return answer
