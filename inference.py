from joint.model import JointModel
from transformers import BertTokenizer, BertModel
import torch
import re
import os
import editdistance
from collections import defaultdict
from tqdm import tqdm

indir = 'demo/in/'
preddir = 'demo/out/'
#compdir = 'saved/demo-out/'

#blocked_comps = set([])

relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
rc2ind = {k: v for v, k in enumerate(relation_classes)}

ner_classes = ['None', 'Name', 'Year', 'Month', 'Day', 'Gender', 'Age']
ner2ind = {k: v for v, k in enumerate(ner_classes)}

model = JointModel(relation_classes, ner_classes)
model.load_state_dict(torch.load('saved/bmodel',map_location=torch.device('cpu')))
model.eval()

model.load_tokenizer()

record = "Le dimanche treize juillet mil huit ce nt vingt huit alheure a midi meme maire et officier de l'etat civil de la commune agiromagny chef lieu canton département du haut Rhin jean Maire etre transporte aneles ant de la maison commune sur luin en nous avous annonce et public a haute deux qu 'il y a promesse du mariage entre Pierre Joseph Temage charpatier age de vingt sept ans gazian majeur au et domicilies agiromagny fils de Pierre demoage & de apoline fagle le pere et mere ramadila giromagny d'unes pous ans Marie Francoise Thomm fille majeure agée de vingt tran ans mei est domicilies avesement canton agiromagny departement du haut Rhin legitime du Maire Shermann cultivateur septrissersule huff les pere et mere domicilies antes vervemont d'antre port cette primaire publication adeduice ete affichee alaporte dela maison commune do nt acte"

with torch.no_grad():

    # correct = 0
    # incorrect = 0
    # extra = 0

    # missed_classes = defaultdict(lambda: 0)
    # extra_classes = defaultdict(lambda: 0)

    files = sorted(os.listdir(indir))
    loop = tqdm(total=len(files))
    for file in files:
        loop.set_description('Processing file...' + file)

        rfile = indir + file
        with open(rfile, 'r') as f:
            record = f.readline()

        preds = model.compute(record)

        if preds is None:
            print('File', file, 'is too long. Skipping.')
            loop.update(1)
            continue

        pfile = preddir + file
        with open(pfile, 'w') as f:
            for text, name in preds:
                f.write(name + '\t' + text + '\n')

        # cfile = compdir + file
        # expected = []
        # with open(cfile, 'r') as f:
        #     for line in f:
        #         name, text = line.strip().split('\t')
        #         if name not in blocked_comps:
        #             expected.append((text, name))

        # for j in range(len(expected)-1, -1, -1):
        #     et, en = expected[j]
        #     for i in range(len(preds)-1, -1, -1):
        #         pt, pn = preds[i]

        #         if en == pn and et == pt:#editdistance.eval(et, pt) / len(et) < 0.2:
        #             #print(i, j, len(preds), len(expected))
        #             del preds[i]
        #             del expected[j]
        #             correct += 1
        #             break
        
        # incorrect += len(expected)
        # extra += len(preds)
        # for exp in expected:
        #     missed_classes[exp[1]]+=1
        # for p in preds:
        #     extra_classes[p[1]]+=1
        loop.update(1)
    loop.close()


    #print(correct, incorrect, extra)
    #print(extra_classes)
    #print(missed_classes)
