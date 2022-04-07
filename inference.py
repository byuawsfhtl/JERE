from joint.model import JointModel
from transformers import BertTokenizer, BertModel
import torch
import re

relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
rc2ind = {k: v for v, k in enumerate(relation_classes)}

ner_classes = ['None', 'Name', 'Year', 'Month', 'Day', 'Gender', 'Age']
ner2ind = {k: v for v, k in enumerate(ner_classes)}

model = JointModel(relation_classes, ner_classes)
model.load_state_dict(torch.load('saved/bmodel-date',map_location=torch.device('cpu')))
model.eval()

model.load_tokenizer()

record = "Le dimanche treize juillet mil huit ce nt vingt huit alheure a midi meme maire et officier de l'etat civil de la commune agiromagny chef lieu canton département du haut Rhin jean Maire etre transporte aneles ant de la maison commune sur luin en nous avous annonce et public a haute deux qu 'il y a promesse du mariage entre Pierre Joseph Temage charpatier age de vingt sept ans gazian majeur au et domicilies agiromagny fils de Pierre demoage & de apoline fagle le pere et mere ramadila giromagny d'unes pous ans Marie Francoise Thomm fille majeure agée de vingt tran ans mei est domicilies avesement canton agiromagny departement du haut Rhin legitime du Maire Shermann cultivateur septrissersule huff les pere et mere domicilies antes vervemont d'antre port cette primaire publication adeduice ete affichee alaporte dela maison commune do nt acte"

with torch.no_grad():
    ans = model.compute(record)

    #for pair in ans:
    #    print(pair)