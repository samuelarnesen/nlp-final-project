#!/usr/bin/python
# -*- coding: utf-8 -*-

# load data
import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn.utils
import torch.nn as nn
import numpy as np
import pandas as pd
import time

import transformers # huggingface
from transformers import BertForSequenceClassification, BertTokenizer, BertModel

pretrained_weights = 'bert-base-uncased' # for bert model


class CustomDataset(Dataset):
    def __init__(self, x, y, num_labels):
        super().__init__()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self._num_labels = num_labels

    def num_labels(self):
        return self._num_labels

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

def get_wiki_data(tokenizer, cutoff=0.3, max_length=512, debugging=False):
    """
    @cutoff (float): cutoff of voter proportion for a comment to be considered abusive
    @max_length (int): maximum length of input (cut off rest of comment)
    """
    comment_df = pd.read_csv("../data/attack_annotated_comments.tsv", sep ='\t')
    comment_df = comment_df.drop(columns=['logged_in', 'ns', 'sample'])
    comment_df["comment"] = comment_df["comment"].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comment_df["comment"] = comment_df["comment"].apply(lambda x: x.replace("TAB_TOKEN", " "))

    annotation_df = pd.read_csv("../data/attack_annotations.tsv",  sep='\t')
    annotation_df = (annotation_df.groupby("rev_id")["attack"].mean() > cutoff)
    annotation_df = annotation_df.to_frame().reset_index()

    final_df = pd.merge(comment_df, annotation_df, how='inner', on=['rev_id'])
    f = lambda s: np.array(tokenizer.encode(s, pad_to_max_length=True, max_length=max_length))

    datasets = {}
    if debugging:
        split = 'dev' # all datasets are dev for speed of implementation
        final_split = final_df[final_df['split'] == split]
        x = np.array(list(final_split['comment'].apply(f).values)) # tokenize!
        y = final_split['attack'].astype(int).values
        d = CustomDataset(x, y, 2) # num_labels = 2
        for k in final_df['split'].unique(): datasets[k] = d
    else:
        for split in final_df['split'].unique(): # train, dev, test
            final_split = final_df[final_df['split'] == split]
            x = np.array(list(final_split['comment'].apply(f).values)) # tokenize!
            y = final_split['attack'].astype(int).values
            datasets[split] = CustomDataset(x, y, 2) # num_labels = 2
    return datasets


def get_fake_data(tokenizer, max_length=512, debugging=False):
    body_df = pd.read_csv("../data/fake_news_bodies.csv")
    stance_df = pd.read_csv("../data/fake_news_stances.csv")

    idx_to_id = {body_id:i for (i, body_id) in enumerate(body_df['Body ID'])}
    stance_to_idx = {stance: i for i, stance in enumerate(stance_df["Stance"].unique())}
    separator = ' ' + tokenizer.sep_token + ' '

    x_list, y_list = [], []
    f = lambda s: np.array(tokenizer.encode(s, pad_to_max_length=True, max_length=max_length))
    count = 0
    for body_id, headline, stance in zip(stance_df["Body ID"], stance_df["Headline"], stance_df["Stance"]):
        body = body_df.iloc[idx_to_id[body_id]]['articleBody']
        text = headline + separator + body
        x_list.append(f(text))
        y_list.append(stance_to_idx[stance])
        count += 1
        if debugging and count > 10:
            break
    x, y = np.array(x_list), np.array(y_list)

    # split dataset
    n = len(x)
    k = int(0.2*n)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    train_ind, dev_ind, test_ind = indices[:-2*k], indices[-2*k:-k], indices[-k:]
    datasets = {}
    datasets['train'] = CustomDataset(x[train_ind], y[train_ind], 4) # num_labels = 4
    datasets['dev'] = CustomDataset(x[dev_ind], y[dev_ind], 4)
    datasets['test_ind'] = CustomDataset(x[test_ind], y[test_ind], 4)
    return datasets


# test get_data method
if __name__ == "__main__":
    print("getting data")
    t_start = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wiki_data = get_wiki_data(tokenizer)
    fake_data = get_fake_data(tokenizer)
    print("success! time taken: {}".format(time.time() - t_start))


