import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class JointModel(nn.Module):
  def __init__(self, re_classes, ner_classes):
    super(JointModel, self).__init__()
    self.bert = BertModel.from_pretrained("./seed/bert", output_hidden_states=True) #"bert-base-multilingual-cased", output_hidden_states=True)
    size = 768 #9984 #768
    dropout = 0.5
    self.norm = nn.Sequential(
        nn.Dropout(dropout),
        nn.LayerNorm(size)
    )
    self.classify_re_left = nn.Linear(size, len(re_classes))
    self.classify_re_right = nn.Linear(size, len(re_classes))
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
        if ner[i][state] * bio[i][1] > 0.25: # continue
          end += 1
        else: # end
          entities.append((start, end, state))
          state = 0
      for j in range(1, num_ner):
        if ner[i][j] * bio[i][2] > 0.25:
          state = j
          start = i
          end = i
          break

    return entities