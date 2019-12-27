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

from dataloader import get_data, Vocabulary, WikiDataset
from models import TransformerClassifier


# create a model with provided arguments
def model_from_dataset(vocab_size, num_labels, args):
    model = TransformerClassifier(vocab_size, num_labels, args['embedding_dim'], args['nheads'], \
        args['layers'], args['feedforward_dim'])
    return model

# split dataset into train, validation, and test
def split_dataset(dataset, train_size, val_size, test_size):
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# create/empty directory to save models in
def new_dir(dir_path):
    assert dir_path != os.getcwd() # should not be creating current directory
    if os.path.isdir(dir_path): 
        print("\twarning, overwriting directory {}".format(dir_path))
        shutil.rmtree(dir_path)
        """
        response = input("do you wish to continue? type yes or no: ")
        if response.lower() == "yes":
            print("deleting existing directory {}".format(dir_path))
            shutil.rmtree(dir_path)
        else:
            sys.exit(0)
        """
    os.makedirs(dir_path) # check valid directory

# evaluate model accuracy and loss on given dataset
def evaluate(model, dataset, num_labels, batch_size=32, verbose=True):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

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
    if verbose: print("evaluation: loss {:5.3f}, accuracy {:5.3f}".format(mean_loss, mean_accuracy))
    return mean_loss, mean_accuracy, predictions, truth

# method for training transformer model on given dataset
def train(vocab, dataset, save_dir, args):

    """ prepare dataset and saving directory """
    new_dir(save_dir)
    n, num_labels = len(dataset), dataset.num_labels()
    n_train, n_val, n_test = n - 2*int(0.15*n), int(0.15*n), int(0.15*n)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, n_train, n_val, n_test)

    """ create model and prepare optimizer """
    model = model_from_dataset(len(vocab), num_labels, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95) # note: 0.95^13 is approx 0.5
    
    """ train model """
    print("starting training")
    train_start = time.time()
    for epoch in range(args['epochs']):
        model.train() # turn on train mode (turned off in evaluate)
        total_loss = 0.
        batch_start = time.time()
        for (x_batch, y_batch) in train_dataloader: # different shuffle each time
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1, num_labels), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad_norm'])
            optimizer.step()
            #scheduler.step()
            total_loss += loss.item()
        total_loss /= len(train_dataset)
        batch_time = time.time() - batch_start
        total_time = time.time() - train_start
        
        # print info from this training epoch
        print('| epoch {:3d} | {:5d} samples | lr {:02.2f} | epoch time {:5.2f} | '
              'total time {:5.2f} | loss {:5.3f}'.format(epoch+1, len(train_dataset), 
                args['lr'], batch_time, total_time, total_loss)) # or scheduler.get_lr()[0]
        evaluate(model, val_dataset, num_labels) # validation
        
        # save model to new directory
        if (epoch+1) % args['save_frequency'] == 0 and epoch+1 != args['epochs']:
            path = os.path.join(save_dir, "epoch-{}".format(epoch+1))
            torch.save(model.state_dict(), path) # store trained model
            
    """ evaluate and save final model """
    print("training complete, evaluating on test dataset")
    model.eval()
    evaluate(model, test_dataset, num_labels)
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
    parser.add_argument('--epochs', type=int, default=25) # epochs to train for
    parser.add_argument('--batch_size', type=int, default=32) # dimension of hidden layers in Actor-Critic neural networks
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--nheads', type=float, default=1)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--feedforward_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=0.5)
    parser.add_argument('--save_frequency', type=int, default=1)
    args = vars(parser.parse_args()) # vars casts to dictionary


    """ prepare dataset """
    valid_datasets = ['wiki', 'fake_news']
    if args['dataset'] == '':
        print("\tno dataset specified, loading all datasets")
    elif args['dataset'] in valid_datasets: 
        print("\tloading dataset {}".format(args['dataset']))
    else:
        raise Exception("{} is not in {}".format(args['dataset'], valid_datasets))
    t_start = time.time()
    vocab, data_dict = get_data()
    wiki_data, fake_news_data = data_dict['wiki'], data_dict['fake news']
    print("\tdatasets loaded, time taken = {}".format(time.time() - t_start))


    """ train models """
    base_dir = os.getcwd() if args['dir_name'] == '' else args['dir_name']
    if args['dataset'] in ['', 'wiki']:
        print("\ttraining model on wikipedia dataset")
        wiki_dir = os.path.join(base_dir, "wiki") 
        train(vocab, wiki_data, wiki_dir, args)
        print("\twiki model complete")

    if args['dataset'] in ['', 'fake_news']:
        print("\ttraining model on fake news dataset")
        fake_dir = os.path.join(base_dir, "fake") 
        train(vocab, fake_news_data, fake_dir, args)
        print("\tfake news model complete")

    print("\ntraining complete\n")





