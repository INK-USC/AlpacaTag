from __future__ import print_function
from torch.autograd import Variable
import time
import sys
import os
import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from utils import *


class Trainer(object):

    def __init__(self, model, optimizer, usecuda=False):
        self.model = model
        self.optimizer = optimizer
        self.usecuda = usecuda

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_model(self, num_epochs, train_data, learning_rate, plot_every=5, adjust_lr=True,
                    batch_size=16, lr_decay = 0.05):

        losses = []
        loss = 0.0
        count = 0
        word_count = 0

        self.model.train(True)
        for epoch in range(1, num_epochs+1):
            train_batches = create_batches(train_data, batch_size= batch_size, order='random')

            for i, index in enumerate(np.random.permutation(len(train_batches))):

                data = train_batches[index]
                self.model.zero_grad()

                words = data['words']
                tags = data['tags']
                chars = data['chars']
                caps = data['caps']
                mask = data['tagsmask']

                if self.usecuda:
                    words = Variable(torch.LongTensor(words)).cuda()
                    chars = Variable(torch.LongTensor(chars)).cuda()
                    caps = Variable(torch.LongTensor(caps)).cuda()
                    mask = Variable(torch.LongTensor(mask)).cuda()
                    tags = Variable(torch.LongTensor(tags)).cuda()
                else:
                    words = Variable(torch.LongTensor(words))
                    chars = Variable(torch.LongTensor(chars))
                    caps = Variable(torch.LongTensor(caps))
                    mask = Variable(torch.LongTensor(mask))
                    tags = Variable(torch.LongTensor(tags))

                wordslen = data['wordslen']
                charslen = data['charslen']

                score = self.model(words, tags, chars, caps, wordslen, charslen, mask)
                loss += score.item()/np.sum(data['wordslen'])
                score.backward()

                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()

                count += 1
                word_count += batch_size

                if count % plot_every == 0:
                    loss /= plot_every
                    print(word_count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0

            if adjust_lr:
                self.adjust_learning_rate(self.optimizer, lr=learning_rate/(1+lr_decay*float(word_count)/len(train_data)))

        return self.model
