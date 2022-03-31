from joint.model import JointModel
from transformers import BertTokenizer, BertModel
import torch
import re

relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
rc2ind = {k: v for v, k in enumerate(relation_classes)}

ner_classes = ['None', 'Name', 'Date', 'Gender', 'Age']
ner2ind = {k: v for v, k in enumerate(ner_classes)}

tokenizer = BertTokenizer.from_pretrained('./seed/tokenizer')#'bert-base-multilingual-cased')
model = JointModel(relation_classes, ner_classes)
model.load_state_dict(torch.load('saved/bmodel',map_location=torch.device('cpu')))

record = "L'an mil sept cent quatre vingt sept le vingt quatre du mois de septembre a jean george vants ruther demeurant au val pavoisse de Rougemont age de quarante huit ans a contracté mariage selon la forme du st concile de trente et les dits royaux avec Marianne cravat fille de jean claude cravat et de sue francoise Bobay de st germain agée de vingt quatre ans en presence de jean claude cravat de jacques cordier jacques montavon soussignés et de jean vantsruther illiterê"

tokens = tokenizer.tokenize(record)

ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

with torch.no_grad():
    ans = model.compute(ids)

    for s, e, i in ans:
        print(' '.join(tokens[s:e+1]).replace(' ##', ''), '=>', ner_classes[i])