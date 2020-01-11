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
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler

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
def evaluate(model, dataset, num_labels, batch_size=32, debugging=False):
    t_start = time.time()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    n = len(dataset)
    predictions = np.zeros(n) # used for confusion matrix
    truth = np.zeros(n)
    total_loss = 0
    curr = 0
    with torch.no_grad():
        for (x, y) in dataloader:
            pred = model(x, labels=y)
            predictions[curr:min(n,curr+batch_size)] = torch.argmax(pred[1], axis=1)
            truth[curr:min(n,curr+batch_size)] = y
            total_loss += pred[0].item()
            curr += batch_size
            if debugging: break # one batch if debugging
    mean_loss = total_loss / n
    mean_accuracy = np.mean(predictions == truth)
    time_taken = time.time() - t_start
    print("evaluation time {}: loss {}, accuracy {}, mean prediction {}".format(time_taken,mean_loss, mean_accuracy, np.mean(predictions)))
    """
    if num_labels == 2: # if binary, print f-beta score
        print("num labels = 2, printing F1 score")
        if np.mean(predictions == 1) == 0 or np.mean(predictions == 0) == 0:
            print("all predictions are {}".format(np.mean(predictions)))
        else:
            precision = np.mean(predictions == truth) / np.mean(predictions == 1)
            recall = np.mean(predictions == truth) / np.mean(truth == 1)
            # higher beta -> prioritize precision, lower beta -> prioritize recall
            f_beta_score = (1+beta**2) * precision * recall / (beta**2 * precision + recall)
            print("precision = {}, recall = {}, f_beta score = {} for beta = {}".format(
                precision, recall, f_beta_score, beta))
    """
    return mean_loss, mean_accuracy, predictions, truth

# method for training transformer model on given dataset
def train(dataset, save_dir, args, balance=False, debugging=False):

    """ prepare dataset and saving directory """
    if not debugging: new_dir(save_dir)
    train, dev, test = dataset['train'], dataset['dev'], dataset['test']
    num_labels = train.num_labels()
    n = len(dataset['train'])

    """ create dataloader """
    samples = SequentialSampler(train)
    if balance:
        frequencies = {}
        for pair in train:
            if pair[1].item() not in frequencies:
                frequencies[pair[1].item()] = 0
            frequencies[pair[1].item()] += 1
        weights = []
        for pair in train:
            weights.append(1/frequencies[pair[1].item()])
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train))
    train_dataloader = DataLoader(train, sampler=sampler, batch_size=args['batch_size'])

    """ create model and prepare optimizer """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    #train_dataloader = DataLoader(train, batch_size=args['batch_size'], shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args['lr'])
    total_steps = len(train_dataloader) * args['epochs'] # number of batches * number of epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    """ train model """
    print("starting training")
    train_start = time.time()
    for epoch in range(1 if debugging else args['epochs']):
        print("\nstarting epoch {} out of {}".format(epoch+1, args['epochs']))
        print("time taken so far: {}\n".format(time.time() - train_start))
        model.train() # turn on train mode (turned off in evaluate)
        total_loss = 0.
        curr = 0
        predictions = np.zeros(args['batch_size'] if debugging else n) # used for confusion matrix
        truth = np.zeros(args['batch_size'] if debugging else n)
        epoch_time = time.time()
        for (x_batch, y_batch) in train_dataloader: # different shuffle each time
            optimizer.zero_grad()
            output = model(x_batch, labels=y_batch)
            loss, preds = output[0], output[1]
            predictions[curr:min(n,curr+args['batch_size'])] = torch.argmax(preds, axis=1)
            truth[curr:min(n,curr+args['batch_size'])] = y_batch
            total_loss += loss.item()
            curr += args['batch_size']
            loss.backward() # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad_norm'])
            optimizer.step()
            scheduler.step()
            if debugging: break # only 1 batch when debugging
            batch_frequent_check = 100 # fraction of a batch at which to log at (recommended 10-20)
            #if (batch_frequent_check*int(curr/args['batch_size'])) % int(n / args['batch_size']) < batch_frequent_check:
            if (int(curr/args['batch_size']) % 10 == 0):
                print("{:4f}% through training".format(100*curr/n))
                print("time taken so far: {}\n".format(time.time() - train_start))

        total_loss /= len(train)
        epoch_time = time.time() - epoch_time
        total_time = time.time() - train_start
        accuracy = np.mean(predictions == truth)
        
        # print info from this training epoch
        print('train - epoch {:3d} | accuracy {:5.5f} | {:5d} samples | lr {:02.3f} | epoch time {:5.2f} | '
              'total time {:5.2f} | mean loss {:5.5f}'.format(epoch+1, accuracy, len(train), 
                scheduler.get_lr()[0], epoch_time, total_time, total_loss))

        evaluate(model, dev, num_labels, batch_size=args['batch_size'], debugging=debugging) # validation
        
        # save model to new directory if not debugging
        if not debugging:
            if (epoch+1) % args['save_frequency'] == 0 and epoch+1 != args['epochs']:
                path = os.path.join(save_dir, "epoch-{}".format(epoch+1))
                new_dir(path)
                model.save_pretrained(path)
            
    """ evaluate and save final model """
    print("training complete, evaluating on test dataset")
    evaluate(model, test, num_labels)
    if not debugging:
        path = os.path.join(save_dir, "final")
        new_dir(path)
        model.save_pretrained(path)
    """ 
    Note: see https://stackoverflow.com/questions/42703500/
    best-way-to-save-a-trained-model-in-pytorch — the best way to save a model
    is to save the state, then to load using
    new_model = TheModelClass(*args, **kwargs)
    new_model.load_state_dict(torch.load(path))
    """
    return model


if __name__ == "__main__":

    print("\nbert_train\n")

    """ parse hyperparameters and other specifications """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
    parser.add_argument('--dataset', type=str, default='', help='either wiki or fake_news') # which dataset to use?
    parser.add_argument('--epochs', type=int, default=3) # usually no more than 1 epochs recommended even
    parser.add_argument('--batch_size', type=int, default=32) # 128 leads to AWS instance running out of memory
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--debug', action='store_true') # debug mode - not training or saving models
    parser.add_argument('--balance', action='store_true') # unbalanced classes
    args = vars(parser.parse_args()) # vars casts to dictionary
    base_dir = os.getcwd() if args['dir_name'] == '' else args['dir_name']

    """ prepare tokenizer """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    """ train models """
    if args['dataset'] in ['', 'wiki']:
        print("\tloading wiki data")
        t_start = time.time()
        dataset = get_wiki_data(tokenizer, debugging=args['debug'])
        print("\twiki data loaded, time taken {}".format(time.time() - t_start))
        print("\ttraining model on wikipedia dataset")
        wiki_dir = os.path.join(base_dir, "wiki") 
        train(dataset, wiki_dir, args, balance=args['balance'], debugging=args['debug'])
        print("\twiki model complete")

    if args['dataset'] in ['', 'fake_news']:
        print("\tloading fake news data")
        t_start = time.time()
        dataset = get_fake_data(tokenizer, debugging=args['debug'])
        print("\tfake data loaded, time taken {}".format(time.time() - t_start))
        print("\ttraining model on fake news dataset")
        fake_dir = os.path.join(base_dir, "fake") 
        train(dataset, fake_dir, args, balance=args['balance'], debugging=args['debug'])
        print("\tfake news model complete")

    print("\ntraining complete\n")



