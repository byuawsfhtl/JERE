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

# todo: take arguments
def load_datasets():
    tokenizer = BertTokenizer.from_pretrained('./seed/tokenizer')#'bert-base-multilingual-cased')

    relation_classes = ['None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf']
    rc2ind = {k: v for v, k in enumerate(relation_classes)}

    ner_classes = ['None', 'Name', 'Year', 'Month', 'Day', 'Gender', 'Age']
    ner2ind = {k: v for v, k in enumerate(ner_classes)}

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
                #add_relation(relations, rev, 'MotherName', e, 'SpouseOf')
            elif e[1] == 'FatherAge':
                add_relation(relations, rev, 'FatherName', e, 'AgeOf')
            elif e[1] == 'SelfGender':
                add_relation(relations, rev, 'SelfName', e, 'GenderOf')
            elif e[1] == 'SpouseGender':
                add_relation(relations, rev, 'SpouseName', e, 'GenderOf')
            elif 'SpouseBirth' in e[1]:
                add_relation(relations, rev, 'SpouseName', e, 'BirthOf')
            elif 'Birth' in e[1]: # not spouse birth
                add_relation(relations, rev, 'SelfName', e, 'BirthOf')
            elif e[1] == 'MotherName':
                add_relation(relations, rev, 'SelfName', e, 'MotherOf')
                #add_relation(relations, rev, 'FatherName', e, 'SpouseOf')
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
                #add_relation(relations, rev, 'SpouseFatherName', e, 'SpouseOf')
            elif e[1] == 'SpouseMotherAge':
                add_relation(relations, rev, 'SpouseMotherName', e, 'AgeOf')
            elif e[1] == 'SpouseFatherAge':
                add_relation(relations, rev, 'SpouseFatherName', e, 'AgeOf')
            elif e[1] == 'SpouseFatherName':
                add_relation(relations, rev, 'SpouseName', e, 'FatherOf')
                #add_relation(relations, rev, 'SpouseMotherName', e, 'SpouseOf')
            elif 'Marriage' in e[1]:
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

            #print(parts)
            #print(parts, cur_entities)

            # Use for picking ner tags
            if 'Name' in parts[1] or 'Person' in parts[1]:
                sub = 'Name'
            elif 'Year' in parts[1]:
                sub = 'Year'
            elif 'Month' in parts[1]:
                sub = 'Month'
            elif 'Day' in parts[1]:
                sub = 'Day'
            elif 'Gender' in parts[1]:
                sub = 'Gender'
            elif 'Age' in parts[1]:
                sub = 'Age'
            else:
                sub = 'None'

            # Use for computing relationships
            parts[1] = parts[1].replace('Surname', 'Name')
            parts[1] = parts[1].replace('GivenName', 'Name')
            parts[1] = parts[1].replace('Person', 'Name')
            #parts[1] = parts[1].replace('Year', 'Date')
            #parts[1] = parts[1].replace('Month', 'Date')
            #parts[1] = parts[1].replace('Day', 'Date')
            if parts[1] == 'Name':
                parts[1] = 'SelfName'
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

    aug = parse_file('out/frenchner.txt') # todo: arg
    test = parse_file('data/all.tsv') # todo: arg

    # relation_classes = set()
    # for record in aug:
    #   relation_classes.update([x[1] for x in record[2]])
    # relation_classes = list(relation_classes)
    # relation_classes.insert(0, 'None')
    
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

    return train_re_dataset_loader, val_re_dataset_loader, test_re_dataset_loader, train_ner_dataset_loader, val_ner_dataset_loader, test_ner_dataset_loader
