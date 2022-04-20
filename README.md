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

The raw text is tokenized and passed through two copies of BERT to produce two sets of embeddings. The first embeddings are all individually passed through two linear layers to generate the NER tags with BIO (Beginning/In/Out) markers.

Unlike PURE, this solution does not use text markers, as they require the text to be reencoded for every pair of entities.

### Testing notes

BERT version/Tokenizer
Joint model
Shared embedding
Decoding
