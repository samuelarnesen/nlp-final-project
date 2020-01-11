#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import sys
import os
import argparse
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bert_dataloader import get_wiki_data, get_fake_data

import transformers # huggingface
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# create/empty directory to save models in
def new_dir(dir_path):
    assert dir_path != os.getcwd() # should not be creating current directory
    if os.path.isdir(dir_path): 
        print("\twarning, overwriting directory {}".format(dir_path))
        shutil.rmtree(dir_path)
    os.makedirs(dir_path) # check valid directory

# evaluate model accuracy and loss on given dataset
def evaluate(model, dataset, num_labels, batch_size=32, verbose=True, beta=1.0): # beta for f-beta score
    model.eval()

    return # TEMPORARY

    dataloader = DataLoader(dataset, batch_size=batch_size)

    n = len(dataset)
    predictions = np.zeros(n) # used for confusion matrix
    truth = np.zeros(n)
    total_loss = 0
    curr = 0
    with torch.no_grad():
        for (x, y) in dataloader:
            pred = model(x)
            predictions[curr:min(n,curr+batch_size)] = torch.argmax(pred, axis=1)
            truth[curr:min(n,curr+batch_size)] = y
            total_loss += criterion(pred.view(-1, num_labels), y).item()
            curr += batch_size
    mean_loss = total_loss / n
    mean_accuracy = np.mean(predictions == truth)
    if verbose: 
        if num_labels == 2: # if binary, print f-beta score
            if np.mean(predictions == 1) == 0 or np.mean(predictions == 0) == 0:
                print("all predictions are {}".format(np.mean(predictions)))
            else:
                precision = np.mean(predictions == truth) / np.mean(predictions == 1)
                recall = np.mean(predictions == truth) / np.mean(truth == 1)
                # higher beta -> prioritize precision, lower beta -> prioritize recall
                f_beta_score = (1+beta**2) * precision * recall / (beta**2 * precision + recall)
                print("precision = {}, recall = {}, f_beta score = {} for beta = {}".format(
                    precision, recall, f_beta_score, beta))
        print("evaluation: loss {:5.5f}, accuracy {:7.5f}".format(mean_loss, mean_accuracy))
    return mean_loss, mean_accuracy, predictions, truth

# method for training transformer model on given dataset
def train(dataset, save_dir, args, debugging=False):

    """ prepare dataset and saving directory """
    if not debugging: 
        new_dir(save_dir)
    train, dev, test = dataset['train'], dataset['dev'], dataset['test']
    num_labels = train.num_labels()

    """ create model and prepare optimizer """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    train_dataloader = DataLoader(train, batch_size=args['batch_size'], shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args['lr'])
    total_steps = len(train_dataloader) * epochs # number of batches * number of epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    """ train model """
    print("starting training")
    train_start = time.time()
    for epoch in range(1 if debugging else args['epochs']):
        model.train() # turn on train mode (turned off in evaluate)
        total_loss = 0.
        epoch_time = time.time()
        for (x_batch, y_batch) in train_dataloader: # different shuffle each time
            optimizer.zero_grad()
            output = model(x_batch, labels=y_batch)
            loss = output[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad_norm'])
            optimizer.step()
            scheduler.step()

        total_loss /= len(train_dataset)
        epoch_time = time.time() - epoch_time
        total_time = time.time() - train_start
        
        # print info from this training epoch
        print('| epoch {:3d} | {:5d} samples | lr {:02.3f} | epoch time {:5.2f} | '
              'total time {:5.2f} | mean loss {:5.5f}'.format(epoch+1, len(train_dataset), 
                scheduler.get_lr()[0], epoch_time, total_time, total_loss))

        evaluate(model, dev, num_labels) # validation
        
        # save model to new directory if not debugging
        if not debugging:
            if (epoch+1) % args['save_frequency'] == 0 and epoch+1 != args['epochs']:
                path = os.path.join(save_dir, "epoch-{}".format(epoch+1))
                torch.save(model.state_dict(), path) # store trained model
            
    """ evaluate and save final model """
    print("training complete, evaluating on test dataset")
    model.eval()
    evaluate(model, test, num_labels)
    if not debugging:
        path = os.path.join(save_dir, "final")
        torch.save(model.state_dict(), path) # store trained model
    """ 
    Note: see https://stackoverflow.com/questions/42703500/
    best-way-to-save-a-trained-model-in-pytorch — the best way to save a model
    is to save the state, then to load using
    new_model = TheModelClass(*args, **kwargs)
    new_model.load_state_dict(torch.load(path))
    """
    return model


if __name__ == "__main__":

    print("\nstart training\n")

    """ parse hyperparameters and other specifications """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
    parser.add_argument('--dataset', type=str, default='', help='either wiki or fake_news') # which dataset to use?
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--debug', action='store_true') # debug mode - not training or saving models
    args = vars(parser.parse_args()) # vars casts to dictionary
    base_dir = os.getcwd() if args['dir_name'] == '' else args['dir_name']

    """ prepare tokenizer """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    """ train models """
    if args['dataset'] in ['', 'wiki']:
        print("\tloading wiki data")
        dataset = get_wiki_data(tokenizer)
        print("\ttraining model on wikipedia dataset")
        wiki_dir = os.path.join(base_dir, "wiki") 
        train(dataset, wiki_dir, args, args['debug'])
        print("\twiki model complete")

    if args['dataset'] in ['', 'fake_news']:
        print("\tloading fake news data")
        dataset = get_fake_data(tokenizer)
        print("\ttraining model on fake news dataset")
        fake_dir = os.path.join(base_dir, "fake") 
        train(dataset, fake_dir, args, args['debug'])
        print("\tfake news model complete")

    print("\ntraining complete\n")



