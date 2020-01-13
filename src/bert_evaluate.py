#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import sys
import os
import argparse
import shutil
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler

from bert_dataloader import get_wiki_data, get_fake_data
from bert_models import BertMultiHeadModel # Custom model 

from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup, PretrainedConfig


def get_preds_truth(model, dataset, early_cutoff=0, batch=32, head=None):
    print("\nrunning get_preds_truth")
    t_start = time.time()
    dataloader = DataLoader(dataset, batch_size=batch)
    n = early_cutoff if early_cutoff > 0 else len(dataset)
    curr = 0
    preds, truth = np.zeros(n), np.zeros(n)
    for (x_batch, y_batch) in dataloader:
        if head == None:
            output = model(x_batch)
        else:
            output = model(head, x_batch)
        preds[curr:min(n, curr+batch)] = torch.argmax(output[0], axis=1)
        truth[curr:min(n, curr+batch)] = y_batch
        curr += batch
        if curr >= n: break
    print("getting preds and truth complete, time taken: {}\n".format(time.time() - t_start))
    return preds, truth

def print_reg_scores(preds, truth, multi_label=False):
    if multi_label:
        print("fake news accuracy: {:5f}".format(np.mean(preds == truth)))
        num_labels = 4
    else:
        print("wiki accuracy: {:5f}".format(np.mean(preds == truth)))
        num_labels = 2
    pred_prop = np.array([np.sum(preds==i) for i in range(num_labels)]) # 4 labels
    pred_prop = pred_prop / np.sum(pred_prop)
    truth_prop = np.array([np.sum(truth==i) for i in range(num_labels)]) # 4 labels
    truth_prop = truth_prop / np.sum(truth_prop)
    print("per class pred vs truth proportion: ", pred_prop, truth_prop)

def print_f1_scores(preds, truth, multi_label=False):
    if multi_label:
        for i, label in enumerate(['unrelated', 'agree', 'disagree', 'discuss']): # 4 classes
            print("label '{}':".format(label))
            precision = np.mean(preds[preds==i] == truth[preds==i])
            recall = np.mean(preds[truth==i] == truth[truth==i])
            f1_score = 2 * precision * recall / (precision + recall)
            print("f1 score: {:5f}, precision: {:5}, recall: {:5f}".format(f1_score, precision, recall))
        print()
    else:
        precision = np.mean(preds[preds==1] == truth[preds==1])
        recall = np.mean(preds[truth==1] == truth[truth==1])
        f1_score = 2 * precision * recall / (precision + recall)
        print("f1 score: {:5f}, precision: {:5}, recall: {:5f}\n".format(f1_score, precision, recall))

def eval_wiki(model, early_cutoff=0, batch=32):
    """ load model and data """
    print("\nevaluating model on wiki dataset\n")
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wiki_data = get_wiki_data(tokenizer)
    test_data = wiki_data['test']

    """ get preds and truth, print results """
    preds, truth = get_preds_truth(model, test_data, early_cutoff=early_cutoff, batch=batch, head=None)
    print_reg_scores(preds, truth, multi_label=False)
    print_f1_scores(preds, truth, multi_label=False)

    """ save results """
    print("\nsaving dict with preds and truth in wiki_pickle\n")
    d = {'preds': preds, 'truth': truth}
    with open('wiki_pickle.p', 'wb') as file:
        pickle.dump(d, file)
    return

def eval_fake(model, early_cutoff=None, batch=32):
    """ load model and data """
    print("\nevaluating model on fake news dataset\n")
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    fake_data = get_fake_data(tokenizer)
    test_data = fake_data['test']

    """ get preds and truth, print results """
    preds, truth = get_preds_truth(model, test_data, early_cutoff=early_cutoff, batch=batch, head=None)
    print_reg_scores(preds, truth, multi_label=True)
    print_f1_scores(preds, truth, multi_label=True)

    """ save results """
    print("\nsaving dict with preds and truth in fake_pickle\n")
    d = {'preds': preds, 'truth': truth}
    with open('fake_pickle.p', 'wb') as file:
        pickle.dump(d, file)
    return

def eval_multi_head(model, dataset='both', early_cutoff=None, batch=32):
    """ load model and data """
    print("\nevaluating multi-head model on datasets\n")
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    """ wiki dataset """
    if dataset in ['both', 'wiki']:
        wiki_data = get_wiki_data(tokenizer)
        wiki_test_data = wiki_data['test']
        """ get results and print evaluation metrics """
        wiki_preds, wiki_truth = get_preds_truth(model, wiki_test_data, early_cutoff=early_cutoff, batch=batch, head=0)
        print("evaluation metrics for multi-head model")
        print_reg_scores(wiki_preds, wiki_truth, multi_label=False)
        print_f1_scores(wiki_preds, wiki_truth, multi_label=False)
        """ save results """
        print("\nsaving dict with preds and truth in multi_pickle\n")
        d = {'preds': wiki_preds, 'truth': wiki_truth}
        with open('multi_wiki_pickle.p', 'wb') as file:
            pickle.dump(d, file)

    """ fake news dataset """
    if dataset in ['both', 'fake']:
        fake_data = get_fake_data(tokenizer)
        fake_test_data = fake_data['test']
        """ get results and print evaluation metrics """
        fake_preds, fake_truth = get_preds_truth(model, fake_test_data, early_cutoff=early_cutoff, batch=batch, head=1)
        print_reg_scores(fake_preds, fake_truth, multi_label=True)
        print_f1_scores(fake_preds, fake_truth, multi_label=True)
        """ save results """
        print("\nsaving dict with preds and truth in multi_pickle\n")
        d = {'preds': fake_preds, 'truth': fake_truth}
        with open('multi_fake_pickle.p', 'wb') as file:
            pickle.dump(d, file)

    return




if __name__ == "__main__":

    print("\nbert_evaluate\n")

    """ parse hyperparameters and other specifications """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
    parser.add_argument('--model', type=str, default='') # fake, wiki, multi
    parser.add_argument('--early_cutoff', type=int, default=0) # debug mode - few epochs
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--dataset', type=str, default='wiki') # wiki, fake, or both
    args = vars(parser.parse_args()) # vars casts to dictionary

    """ check inputs """
    if args['dir_name'] == '' or not os.path.isdir(args['dir_name']): 
        raise Exception("dir_name must be valid model directory")
    if args['model'] not in ['wiki', 'fake', 'multi']:
        raise Exception("model_type must be specified, either 'fake', 'wiki' or 'multi")

    """ run evaluation """
    if args['model'] == 'wiki':
        model = BertForSequenceClassification.from_pretrained(args['dir_name'])
        eval_wiki(model, early_cutoff=args['early_cutoff'], batch=args['batch_size'])
    if args['model'] == 'fake':
        model = BertForSequenceClassification.from_pretrained(args['dir_name'])
        eval_fake(model, early_cutoff=args['early_cutoff'], batch=args['batch_size'])
    if args['model'] == 'multi':
        config_name = PretrainedConfig().from_json_file(os.path.join(args['dir_name'], "config.json"))
        model = BertMultiHeadModel.from_pretrained(os.path.join(args['dir_name'], "pytorch_model.bin"), config=config_name)
        eval_multi_head(model, dataset=args['dataset'], early_cutoff=args['early_cutoff'], batch=args['batch_size'])

    print("\nbert_evaluate complete\n")






