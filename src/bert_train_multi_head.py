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
from bert_models import BertMultiHeadModel # Custom model 
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup


# create/empty directory to save models in
def new_dir(dir_path):
    assert dir_path != os.getcwd() # should not be creating current directory
    if os.path.isdir(dir_path): 
        print("\twarning, overwriting directory {}".format(dir_path))
        shutil.rmtree(dir_path)
    os.makedirs(dir_path) # check valid directory

# evaluate model accuracy and loss on given dataset
def evaluate(model, wiki_data, fake_data, batch_size=32, debugging=False):
    t_start = time.time()
    model.eval()
    wiki_loader = DataLoader(wiki_data, batch_size=batch_size)
    fake_loader = DataLoader(fake_data, batch_size=batch_size)
    for name, loader, n in zip(['wiki', 'fake'], [wiki_loader, fake_loader], [len(wiki_data), len(fake_data)]):
        predictions = np.zeros(n) # used for confusion matrix
        truth = np.zeros(n)
        total_loss = 0
        curr = 0
        with torch.no_grad():
            for (x, y) in loader:
                head = 0 if name == 'wiki' else 1 # which model head to use
                pred = model(head, x, labels=y)
                predictions[curr:min(n,curr+batch_size)] = torch.argmax(pred[1], axis=1)
                truth[curr:min(n,curr+batch_size)] = y
                total_loss += pred[0].item()
                curr += batch_size
                if debugging: break # one batch if debugging
        mean_loss = total_loss / n
        mean_accuracy = np.mean(predictions == truth)
        time_taken = time.time() - t_start
        print("evaluation for " + name)
        print("time {}: loss {}, accuracy {}, mean prediction {}".format(
            time_taken, mean_loss, mean_accuracy, np.mean(predictions)))

def create_sampler(train): # train == dataset
    frequencies = {}
    for pair in train: # pair = (x, y)
        if pair[1].item() not in frequencies:
            frequencies[pair[1].item()] = 0
        frequencies[pair[1].item()] += 1
    weights = []
    for pair in train:
        weights.append(1/frequencies[pair[1].item()])
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train))
    return sampler

# method for training transformer model on given dataset
def train_multi_head(wiki_data, fake_data, save_dir, args, balance=False, debugging=False):

    """ prepare dataset and saving directory """
    if not debugging: new_dir(save_dir)
    wiki_train, wiki_dev, wiki_test = wiki_data['train'], wiki_data['dev'], wiki_data['test']
    fake_train, fake_dev, fake_test = fake_data['train'], fake_data['dev'], fake_data['test']
    wiki_num_labels, fake_num_labels = wiki_train.num_labels(), fake_train.num_labels()
    wiki_n, fake_n = len(wiki_train), len(fake_train)
    n = min(wiki_n, fake_n) * 2

    """ create dataloader and model """
    wiki_sampler = create_sampler(wiki_train) if balance else SequentialSampler(wiki_train)
    wiki_dataloader = DataLoader(wiki_train, sampler=wiki_sampler, batch_size=args['batch_size'])
    fake_sampler = create_sampler(fake_train) if balance else SequentialSampler(fake_train)
    fake_dataloader = DataLoader(fake_train, sampler=fake_sampler, batch_size=args['batch_size'])

    model = BertMultiHeadModel.from_pretrained("../multi-epoch-6")
    optimizer = AdamW(model.parameters(), lr=args['lr'])
    total_steps = (len(wiki_dataloader) + len(fake_dataloader)) * 2 # number of batches * number of epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    """ train model """
    print("starting training")
    train_start = time.time()
    last_save_time = time.time() # time since last save
    last_save_count = 0
    for epoch in range(1 if debugging else args['epochs']):
        print("\nstarting epoch {} out of {}".format(epoch+1, args['epochs']))
        model.train() # turn on train mode (turned off in evaluate)
        total_loss = 0.
        curr = 0
        wiki_accuracies = []
        fake_accuracies = []
        epoch_time = time.time()
        for (x_wiki, y_wiki), (x_fake, y_fake) in zip(wiki_dataloader, fake_dataloader): # different shuffle each time
            optimizer.zero_grad()
            # train on wiki
            output = model(0, x_wiki, labels=y_wiki) # 0 => wiki head
            loss, preds = output[0], output[1]
            wiki_accuracies.append(np.mean(np.array(torch.argmax(preds, axis=1) == y_wiki)))
            total_loss += loss.item()
            curr += args['batch_size']
            loss.backward() # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad_norm'])
            optimizer.step()
            scheduler.step()
            # train on fake news
            output = model(1, x_fake, labels=y_fake) # 1 => fake news head
            loss, preds = output[0], output[1]
            fake_accuracies.append(np.mean(np.array(torch.argmax(preds, axis=1) == y_fake)))
            total_loss += loss.item()
            curr += args['batch_size']
            loss.backward() # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad_norm'])
            optimizer.step()
            scheduler.step()
            if debugging: break # only 1 batch when debugging
            if time.time() - last_save_time > 3600: # 3600 seconds = 1 hour
                last_save_time = time.time()
                last_save_count += 1
                print("{:4f}% through training".format(100*curr/n))
                print("time taken so far: {}".format(time.time() - train_start))
                print("mean accuracy so far (wiki, fake): {}, {}\n".format(np.mean(wiki_accuracies), np.mean(fake_accuracies)))
                #if not debugging:
                path = os.path.join(save_dir, "save-count-{}".format(last_save_count))
                new_dir(path)
                model.save_pretrained(path)

        total_loss /= n
        epoch_time = time.time() - epoch_time
        total_time = time.time() - train_start
        accuracy = (np.mean(wiki_accuracies) + np.mean(fake_accuracies)) / 2
        
        # print info from this training epoch
        print('train - epoch {:3d} | accuracy {:5.5f} | {:5d} samples | lr {:02.3f} | epoch time {:5.2f} | '
              'total time {:5.2f} | mean loss {:5.5f}'.format(epoch+1, accuracy, n, 
                scheduler.get_lr()[0], epoch_time, total_time, total_loss))

        evaluate(model, wiki_dev, fake_dev, batch_size=args['batch_size'], debugging=debugging) # validation
        
        # save model to new directory if not debugging
        if not debugging:
            if (epoch+1) % args['save_frequency'] == 0 and epoch+1 != args['epochs']:
                path = os.path.join(save_dir, "epoch-{}".format(epoch+1))
                model.save_pretrained(path)
            
    """ evaluate and save final model """
    print("training complete, evaluating on test dataset")
    evaluate(model, wiki_test, fake_test)
    if not debugging:
        path = os.path.join(save_dir, "final")
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

    print("\nbert_train_multi_head\n")

    """ parse hyperparameters and other specifications """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
    parser.add_argument('--dataset', type=str, default='', help='either wiki or fake_news') # which dataset to use?
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32) # max possible
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--debug', action='store_true') # debug mode - not training or saving models
    parser.add_argument('--balance', action='store_true') # unbalanced classes
    args = vars(parser.parse_args()) # vars casts to dictionary
    base_dir = os.getcwd() if args['dir_name'] == '' else args['dir_name']

    """ prepare dataset """
    print("\tloading datasets")
    t_start = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wiki_data = get_wiki_data(tokenizer, debugging=args['debug'])
    fake_data = get_fake_data(tokenizer, debugging=args['debug'])
    print("\tdatasets loaded, time taken: {}".format(time.time() - t_start))

    """ train model """
    print("\ttrain model")
    base_dir = os.getcwd() if args['dir_name'] == '' else args['dir_name']
    model_save_dir = os.path.join(base_dir, "multi-head") 
    train_multi_head(wiki_data, fake_data, model_save_dir, args, args['balance'], args['debug'])

    print("\ntraining complete\n")



