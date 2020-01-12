#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import time

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, TensorDataset
import transformers # huggingface
from transformers import BertModel, BertTokenizer, AdamW, BertPreTrainedModel


class BertMultiHeadModel(BertPreTrainedModel): # FAKE SEQUENCE MODEL
    def __init__(self, config):
        super(BertMultiHeadModel, self).__init__(config)
        self.num_labels = [2, 4] # ignore config.num_labels # should be a list!
        self.num_tasks = 2 # CUSTOM EDIT: MANUALLY SPECIFIED NUM_TASKS

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = [nn.Linear(config.hidden_size, self.num_labels[i]) for i in range(self.num_tasks)]

        self.init_weights()

    def forward(
        self,
        task,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        if type(task) != int:
            raise Exception("BertMulti model first input must be task index (int)!")
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier[task](pooled_output) # CUSTOM EDIT: specify which linear layer with task index

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels[task]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

