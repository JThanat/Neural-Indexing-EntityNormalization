import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import math
import time
import copy
import os
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, BertModel
from transformers import AdamW, BertConfig

from sentence_transformers import SentenceTransformer, models, SentencesDataset, InputExample, losses, evaluation

from datetime import date
import logging
import nltk
import glob

from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import torch
from annoy import AnnoyIndex


def get_data_from_path(path_to_uniprot, seed=124):
    random.seed(seed)

    train_df = pd.read_csv(os.path.join(path_to_uniprot, 'train.tsv'), delimiter='\t').dropna()
    dev_df = pd.read_csv(os.path.join(path_to_uniprot, 'dev.tsv'), delimiter='\t').dropna()
    train_df['name1'] = train_df['name1'].apply(lambda x: x.strip())
    train_df['name2'] = train_df['name2'].apply(lambda x: x.strip())

    dev_df['name1'] = dev_df['name1'].apply(lambda x: x.strip())
    dev_df['name2'] = dev_df['name2'].apply(lambda x: x.strip())
    
    train_data = [list(a) for a in zip(train_df['name1'], train_df['name2'], train_df['label'])]
    dev_data = [list(a) for a in zip(dev_df['name1'], dev_df['name2'], dev_df['label'])]
    
    random.shuffle(train_data)
    random.shuffle(dev_data)
    
    return train_data, dev_data

def generate_heuristic(original_list):
    rows = []
    for name in original_list:
        name = name.strip()
        if len(name) < 10:
            processed_name = set()
            nlower = name.lower()
            nfirst_upper = nlower[0].upper() + nlower[1:]
            
            rows.append([name, nlower, 0.9])
            rows.append([name, nfirst_upper, 0.9])
            
            processed_name.add(nlower)
            processed_name.add(nfirst_upper)
            
            to_compare_name = []
            alnum_name = "".join([c for c in name if c.isalnum()])
            to_compare_name.append(alnum_name)
            to_compare_name.append(alnum_name.lower())
            to_compare_name.append(alnum_name.lower()[0].upper() + alnum_name.lower()[1:])
            for comparing_name in to_compare_name:
                dist = nltk.distance.edit_distance(name, comparing_name)
                sim = 1 - dist * 0.05
                if sim >= 0.8 and comparing_name not in processed_name:
                    rows.append([name, comparing_name, sim])
    return rows


# Create Word Embedding model with max_seq_length of 256
def get_model(mname=None, device='cpu'):
    if not mname:
        # Read biobert from hugging face
        word_embedding_model = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device=device)
        model.to(device)
    else:
        model = SentenceTransformer(mname)
        # saved model tokenizer doesn't seem to work well. Further inspection need
        # thus we directly load pretrained tokenizer
        dmis_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1', model_max_length=256)
        model.tokenizer = dmis_tokenizer
        model.to(device)
    return model


def get_train_dataloader(model, train_data, batch_size=64):
    # Create Train Dataset using InputExample Provided by Sentence Transformer Library
    examples = []
    for i, data in enumerate(train_data):
        s1, s2, score = data
        ex = InputExample(texts=[s1,s2],label=score)
        examples.append(ex)
        
    train_dataset = SentencesDataset(examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    return train_dataloader


def get_dev_dataloader(model, dataset, evaltype='cosine', show_progress_bar=True):
    # Create Binary Classification Output for Dev Set
    s1_list = []
    s2_list = []
    lbl_list = []
    for i, data in enumerate(dataset):
        s1, s2, score = data
        s1_list.append(s1)
        s2_list.append(s2)
        lbl_list.append(score)
    
    if evaltype == 'cosine':
        evaluator = evaluation.EmbeddingSimilarityEvaluator(s1_list, s2_list, lbl_list, show_progress_bar=show_progress_bar)
        return evaluator
    elif evaltype == 'binary':
        evaluator = evaluation.BinaryClassificationEvaluator(s1_list, s2_list, lbl_list, show_progress_bar=show_progress_bar)
        return evaluator
    else:
        evaluator = evaluation.BinaryClassificationEvaluator(s1_list, s2_list, lbl_list, show_progress_bar=show_progress_bar)
        return evaluator

def evaluate_classification_performance(mname, evaluator, output_path=None):
    if output_path and not os.path.exists(output_path):
        os.mkdir(output_path)
    model = get_model(mname, device)
    avg_prec = evaluator(model, output_path=output_path)
    return avg_prec