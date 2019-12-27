import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import sys
import os
import argparse

from dataloader import get_data, Vocabulary, WikiDataset
from models import TransformerClassifier


# create a model of specified dimension
def model_from_dataset(vocab, dataset, args):
    vocab_size, num_labels = len(vocab), dataset.num_labels()
    model = TransformerClassifier(vocab_size, num_labels, args.embedding_dim, args.nheads, \
        args.layers, args.feedforward_dim)
    return model


# method for training transformer model on given dataset
def train(vocab, dataset, save_dir, args):
    assert os.path.isdir(save_dir) # check valid directory
    model = model_from_dataset(vocab, dataset, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.train() # Turn on the train mode

    for epoch in range(args.epochs):
        total_loss = 0.
        start_time = time.time()
        for data in dataloader: # different shuffle each time
            x_batch, y_batch = data
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(args.batch_size, -1), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            total_loss += loss.item()
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} samples | '
              'lr {:02.2f} | time {:5.2f} | loss {:5.2f}'.format(epoch, i, 
                len(train_data), scheduler.get_lr()[0], elapsed, total_loss))
        if (epoch+1) % save_frequency == 0 and epoch+1 != epochs:
            path = os.path.join(save_dir, "/epoch-{}".format(epoch))
            torch.save(model.state_dict(), path) # store trained model
    path = os.path.join(save_dir, "/final")
    """ 
    Note: see https://stackoverflow.com/questions/42703500/
    best-way-to-save-a-trained-model-in-pytorch — the best way to save a model
    is to save the state, then to load using
    new_model = TheModelClass(*args, **kwargs)
    new_model.load_state_dict(torch.load(path))
    """
    torch.save(model.state_dict(), path) # store trained model


if __name__ == "__main__":

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
    args = parser.parse_args()


    """ prepare dataset """
    valid_datasets = ['wiki', 'fake_news']
    if args.dataset == '':
        print("\nno dataset specified, loading all datasets\n")
    else if args.dataset in valid_datasets: 
        print("\nloading dataset {}\n".format(args.dataset))
    else:
        raise Exception("{} is not in {}".format(args.dataset, valid_datasets))
    
    t_start = time.time()
    vocab, data_dict = get_data()
    wiki_data, fake_news_data = data_dict['wiki'], data_dict['fake news']
    print("\ndatasets loaded, time taken = {}".format(time.time() - t_start))


    """ train models """
    base_dir = os.getcwd() if args.dir_name == '' else args.dir_name
    if args.dataset in ['', 'wiki']:
        print("training model on wikipedia dataset\n")
        t_start = time.time()
        wiki_dir = os.path.join(base_dir, "/wiki/") 
        train(vocab, wiki_data, wiki_dir, args)
        print("\nwiki model trained, time taken = {}".format(time.time() - t_start))

    if args.dataset in ['', 'fake_news']:
        print("training model on fake news dataset\n")
        t_start = time.time()
        fake_dir = os.path.join(base_dir, "/fake/")
        train(vocab, fake_news_data, fake_dir, args)
        print("\fake news model trained, time taken = {}".format(time.time() - t_start))

    print("\ntraining complete\n")





