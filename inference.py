from joint.model import JointModel
from transformers import BertTokenizer, BertModel
import torch
import re

relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
rc2ind = {k: v for v, k in enumerate(relation_classes)}

ner_classes = ['None', 'Name', 'Year', 'Month', 'Day', 'Gender', 'Age']
ner2ind = {k: v for v, k in enumerate(ner_classes)}

tokenizer = BertTokenizer.from_pretrained('./seed/tokenizer')#'bert-base-multilingual-cased')
model = JointModel(relation_classes, ner_classes)
model.load_state_dict(torch.load('saved/bmodel-split2',map_location=torch.device('cpu')))
model.eval()

record = "Làn mil huit cent quarante huit le vingt deux Avril à huit heures du matin , pardevant nous Jean Baptiste Lamboley Maire officier de l' Etat civil de la commune de Giromagny Cheflien de Canton , Arrondissement de Belfort , Département du haut Ehin en comparu Jean Baptiste Mouge not li perand agé de trente cing ans domicilié en cette commune . lequel nous a présenté un enfant du séxe masculin quél déclaré être né cejourd ' hui é une heure du matin en son domiciles auguartier du hautôt de cette Commune ddelui déclarant et de Marie Francois Thomme son époux sans profession agée de trente trois ans domiciliée en cette Commune et au quel il a déclaré donne les prénoms de Charles Jean Baptiste . Lesquelles présentation et déclarations faites en présence de Joseph Honard employé de burean agé de cinquante cing ans et d' Alexandre Honard coutreona être de dévid âge agé de trente un ans les deux temoins domiciliés en cette commune Etout le comparant les temoins sus dénommes signé avec nous le présent acte immiédiatement aprés que nous en avons donné lecture"

tokens = tokenizer.tokenize(record)

ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

with torch.no_grad():
    ans = model.compute(ids)

    for s, e, i in ans:
        print(' '.join(tokens[s:e+1]).replace(' ##', ''), '=>', i)