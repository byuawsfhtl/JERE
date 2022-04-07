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

record = "L'an mil huit cent soixante treize le six Octobre a neuf heures du matin pardevant nous Charles Prenez maire et officier de L'Etat civil de la commune de Mont bouton canton de Delle arrondissement de Belfort Département du Haut Rhin est comparu Marguerite Rayot sage femme âgée de trente neuf ans domicilice a Vandoncourt laquelle nous a présente un enfant du sexe masculin qu'elle déclare etre né dans la maison de Francois Contesse en cette commune cejourd ' hui a six heures du matin de Francois Scherrer actuellement militaire âgé de vingt cing ans et de Madeleine Philippe son epouse ménagere âgée de trente ans domicilicé en cette commune et auquel elle a declare vouloir donner les prenoms de Francois Havier lesquelles présentation et déclaration faites en présence de Havier Valles ouvrier de fabrique âgé de vingt un ans et de Henri Monnier Instituteur primaire âgé de cinquante ans les deux domicilies a Montbouton temoins choisis par la déclarant et ont la déclarante et les témoins signé avec nous le présente acte de naissance apres qu'il leur en a ete fait lecture"

with torch.no_grad():
    ans = model.compute(record)

    #for pair in ans:
    #    print(pair)