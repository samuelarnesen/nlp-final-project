from torch.utils.data import Dataset, TensorDataset
import torch.nn.utils
import torch.nn as nn
import numpy as np
import pandas as pd
import os.path

class Vocabulary:

    """
    Class for converting words to indexes and back out again
    """

    def __init__(self, series_list):
        """
        Stores list to map an index number to a word and a dictionary to map a word to an index number

        @param series_list (list(pd.Series)): list of pandas series to be used to form vocabulary
        """

        vocab_dict = dict()
        def add_entry_if_necessary(entry):
            words = entry.split()
            for word in words:
                if word.lower() not in vocab_dict:
                    vocab_dict[word.lower()] = True

        for series in series_list:
            series.apply(add_entry_if_necessary)

        self.index_to_word = list(vocab_dict.keys())
        self.index_to_word.insert(0, "<pad>")
        self.index_to_word.insert(1, "<unk>")
        self.index_to_word.insert(2, "<s>")
        self.index_to_word.insert(3, "</s>")

        self.word_to_index = dict()
        for idx in range(len(self.index_to_word)):
            self.word_to_index[self.index_to_word[idx]] = idx


    def __getitem__(self, item):
        """
        Allows Vocabulary object to be subscripted

        @param item (int or string): if int, gets the word found at that index
                                     if string, gets the index associated with that word

        """
        if type(item) == type("string"):
            return self.word_to_index[item]
        else:
            return self.index_to_word[item]

    def __len__(self):
        return len(self.index_to_word)

    def get_index_list_from_sentence(self, sentence):

        """
        Converts a sentence into a list of indices

        @param sentence (str): a string of words that is to be converted

        @returns idx_list (List(int)): a list of integers where each element corresponds to the index of a word in the input sentence
        """
        idx_list = list()
        idx_list.append(self["<s>"])
        for word in sentence.split():
            if word.lower() not in self.word_to_index:
                idx_list.append(self["<unk>"])
            else:
                idx_list.append(self[word.lower()])
        idx_list.append(self["</s>"])
        return idx_list

    def get_tensor_from_sentences(self, sentences, device: torch.device):

        """
        Makes a torch tensor from a batch of sentences

        @param sentences (List(List(int))): the sentences that will comprise the tensor
        @param device (torch.device): device code is being run on 

        @returns tensor (torch.tensor): padded tensor of input sentences 
        """

        max_length = max(len(sentence) for sentence in sentences)

        word_idxs = list()
        for sentence in sentences:
            word_idxs.append(sentence)
            while len(word_idxs[-1]) < max_length:
                word_idxs[-1].append(self["<pad>"])

        return torch.t(torch.tensor(word_idxs, dtype=torch.long, device=device))


class WikiDataset(Dataset):

    """
    Class for storing input data from Wikipedia dataset
    """

    def __init__(self, comment_df: pd.DataFrame, annotation_df: pd.DataFrame, vocab: Vocabulary):

        """
        @param comment_df (pd.DataFrame): pandas DataFrame with "comments" section that is used as the input
        @param annotation_df (pd.DataFrame): pandas DataFrame that stores the labels
        @param vocab (Vocabulary): vocabulary to be used 
        """

        super().__init__()

        cleaned_comment_df = comment_df.copy()
        cleaned_comment_df["comment"] = cleaned_comment_df["comment"].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        cleaned_comment_df["comment"] = cleaned_comment_df["comment"].apply(lambda x: x.replace("TAB_TOKEN", " "))

        self.vocab = vocab
        self.x = cleaned_comment_df["comment"].apply(self.vocab.get_index_list_from_sentence).values
        self.x = np.array([np.array(x) for x in self.x])
        self.y = (annotation_df[annotation_df["rev_id"].isin(cleaned_comment_df["rev_id"])].groupby("rev_id")["attack"].mean() > 0.5).values
        self.y = np.array([int(i) for i in self.y])
        self.y = np.array([np.array([int(i == j) for i in range(2)]) for j in self.y]) # 2 classes because binary classification

    def __getitem__(self, index):

        """
        @returns (tuple(List(int), bool)): first term is a list of the indices of the words of the input sentence at the specified index
                                           second term is boolean corresponding to whether it is an attack (True) or not (False)
        """
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

class FakeNewsDataset(Dataset):

    def __init__(self, body_df: pd.DataFrame, stance_df: pd.DataFrame, vocab: Vocabulary):

        super().__init__()
        self.vocab = vocab

        stance_to_idx = {}
        stances = stance_df["Stance"].drop_duplicates().values
        for i, stance in enumerate(stances):
            stance_to_idx[stance] = i
        num_stances = len(stance_to_idx)

        body_df["sentence_as_idx"] = body_df["articleBody"].apply(self.vocab.get_index_list_from_sentence)

        x_list = []
        y_list = []
        for row in stance_df.iterrows():
            idx = row[1]["Body ID"]
            headline = self.vocab.get_index_list_from_sentence(row[1]["Headline"])
            body = body_df[body_df["Body ID"] == idx].iloc[0]["sentence_as_idx"]
            x_list.append(np.concatenate([headline, body]))

            y_val = np.zeros(num_stances)
            y_val[stance_to_idx[row[1]["Stance"]]] = 1
            y_list.append(y_val)

        self.x = np.array(x_list)
        self.y = np.array(y_list)

    def __getitem__(self, index):

        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


"""
Description: returns vocab and a dict of all loaded datasets
"""
def get_data():
    comment_df = pd.read_csv("../data/attack_annotated_comments.tsv", sep ='\t')
    body_df = pd.read_csv("../data/fake_news_bodies.csv")
    stance_df = pd.read_csv("../data/fake_news_stances.csv")
    vocab = Vocabulary([comment_df["comment"], body_df["articleBody"], stance_df["Headline"]])
    annotation_df = pd.read_csv("../data/attack_annotations.tsv",  sep='\t')

    wiki_dataset = WikiDataset(comment_df, annotation_df, vocab)
    fake_news_dataset = FakeNewsDataset(body_df, stance_df, vocab)

    return vocab, {"wiki": wiki_dataset, "fake news": fake_news_dataset}
    """
    train_df = comment_df[comment_df["split"] == "train"]
    dev_df = comment_df[comment_df["split"] == "dev"]
    test_df = comment_df[comment_df["split"] == "test"]

    train_data = WikiDataset(train_df, annotations_df)
    """


