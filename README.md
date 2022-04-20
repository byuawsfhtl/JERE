# BMD NER/RE

## Install Dependancies

`environment.yml` has the exported conda environment for running this repository on linux without a GPU, which allows the inference script to be run on the CPU. For training, install the relavant CUDA version of PyTorch instead.

## Running Inference

This repository is provided as a proof of concept, and as such many values are currently hard coded.

To perform inference with a trained model, edit the input, output, and model weights path in `inference.py`. The script loads the saved weights onto a CPU-based model. It takes as input a folder where all of the records are their own files, and it outputs files of the same name to the output folder.

It assumes that the input files have the entire record on one line. The script will skip over long files, as BERT-based models can only handle 512 tokens as input. It outputs a tsv of label -> text containing the post processed extracted entities from the text.

## Training

This repository relys on artificial record generation for data augmentation. The `get-gen.py` script will download and unpack the required code into the currentl directory. As this is a proof of concept, we did not create seperate packages for the different functions.

Place the human-labeled training data in `data/all.tsv`, where each line contains one word a tab and the tag for each word. Delineate records with the line `None none`.

Currently supported NER classes are:
'None', 'Name', 'Year', 'Month', 'Day', 'Gender', 'Age'

Currently supported RE classes are:
'None', 'GenderOf', 'AgeOf', 'FatherOf', 'MotherOf', 'SpouseOf', 'BirthOf', 'MarriageOf'

These labels are inferred from the more fine grain labels contained in the training data. A sample file will be provided as an example.

Run `re_test.py` to train the model. The training loop will train NER first, then RE. The validation loop will check accuracy on both tasks each epoch.

## Methodolgy

### Model Achitecture

The model achitecture was inspired by [PURE](https://github.com/princeton-nlp/PURE).

The raw text of the entire record is tokenized and passed through two copies of BERT to produce two sets of embeddings. The first embeddings are all individually passed through two linear layers to generate the NER tags with BIO (Beginning/In/Out) markers.

The second set of BERT embeddings are used for the RE task. Once the raw text has been processed by BERT and an ordered pair of entities have been identified, the embeddings for the left and right entities are processed by seperate linear layers, and the dot product of their outputs is ran through a linear classifier. This was inspired by self-attention, and it allows an interaction between the selected entities without using text markers in the raw text.

This allows all of the record to be processed in two forward passes through BERT, which is significantly faster than running BERT repeatedly for each pair of entities with different text markers added to the raw text.

### Training

Training examples consisted of a record with only entity or relationship marked. This was done to simplify the training loop and allow for greater variety in the sentences seen each step.

In both tasks, the None class overpowered the positive classes. To balance the classes and have more representative training mini-batches, we subsampled the frequency of the none class to be three times the frequency of the next most popular class.

We employed gradient accumulation over 25 training examples to allow for larger batches than could be processed by the GPU at once.

Both NER and RE were trained seperately and with early stopping. The best model was taken after 5 epochs of not seeing an improvement.

Once NER was done training, the BERT model for RE was initialized with weights transfered from the NER task.

The Adam optimizer was used with CrossEntropyLoss for both tasks.

### Post Processing

The provided post processing step takes in a full record, and classifies the entire record at once.

It identifies spans of tokens as likely entities. It then iterates through all pairs of likely entities and finds the output of the RE task for that pair.

RE output is masked given the class of both entities (i.e. a date cannot be the mother of a person). The relationships are sorted by type such that the main person in the record is identified first, and the tags for other people branch off of their connection to the central figure (e.g. spouses are processed before parents, so that the parents-in-law can be immediately identified as such).

**One of the largest improvements seen in the post processing step was caused by limiting relationships to be only the most likely within a record.** For example, each person in the record can only have one father, and each person will only ever be the father of one person (generally true in the records we are given). This change significantly reduced the number of errors we saw caused by false positives if we accepted all relationships above a threshold of confidence.

### Testing notes

We tried several version of BERT, and noticed the best performance with bert-based-multilingual-cased. AlBERT models performed poorly, which we theorize was due to the repeated attention layer not capturing all required information. It would be worth it to attempt to train/fine-tune a BERT model on the french data that we have and cut the vocabulary size of the tokenizer to be specific to French.

For the RE task, we initially tried concatenating the embeddings produced for each word and running them through a classifier as done in PURE. This performed well, but in the absense of markers, there was no interaction between the selected pair of entities. Due to this, the model had a high false positive rate where it would cross entities from disjoint relationship tuples of the same type. For example, a person that was identified as the spouse of one person in the record would also be identified as the spouse of every other person in the record as well, as long as the non-spouse had a spouse mentioned in the record. This was the reason that we added sudo self-attention before the classifier for the RE task.

Additionally, we also performed several tasks with a fully joint model, where only one BERT model was used, and the produced embeddings were shared between tasks. We saw relatively good performance before using making the change mentioned in the last paragraph. After adding the sudo-attention, however, training the joint model caused the accuracy of the NER and BIO tasks to drop. Because of this, we split the model so that it had two seperate BERT-based embeddings. We theorize that the different tasks use the self-attention in BERT differently, and propose that it would be worth while to attempt sharing the first few layers of BERT and spliting the embeddings afterward. This would increase the speed of inference.

Also, though we trained BIO identification as a proof of concept for named entities, the current post processing step does not make use of it. In the training data that we were given, there was a relatively small amount of cases where the record had two of the same entity immediatly next to each other. Training data where this is the case would be required to adequately train this system. Both the pre and post processing tasks would then have to be modified to take advantage of the change.

The current model only identifies the type of record by the relationship between the main person of the record and the date mentioned in the record. It would likely be helpful to have a seperate model (even a statistical one) predict the type of record, as there are many textual clues beyond the date's relationship with the central figure.

The main error occuring in the model still is recognizing surnames vs given names. The model was not trained on this task, as we believe the most effective way to do that would be to treat the names as two seperate entities during NER, but as one entity during RE. The code base was not written to handle such a case, and we have not redesigned it due to time contraints. The current surname recognition is done with some basic post processing logic (last word, assuming the word 'given name' does not occur prior to the name, and that the name is more than one word of non-trivial length). However, the provided records are not consistent in mentioning the given names before the surname, frequently even within one record. We believe that an NLP model or dictionary approach would be best suited for identifying names. Additionally, much of the training data confuses profession and surname (due to shared vocabulary), which leads the model to frequently drop the surname as a recognized entity entirely.

During the decoding task of post processing, it might be worth it to add temperature to the soft max on outputs. There were many high-scoring false positives due to inherent bias in the models. We believe that this could be partially corrected by a well-picked tempurature applied to the softmax.