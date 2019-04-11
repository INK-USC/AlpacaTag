import numpy as np
import time
from utils import *

class Acquisition(object):

    #currently train_data = [['EU','rejects','it',,,,],]
    def __init__(self, train_data, init_percent=2, seed=0):
        self.tokenlen = sum([len(x) for x in train_data])
        self.train_index = set()
        self.obtain_data(train_data, acquire=init_percent)

    def obtain_data(self, data, model, acquire=2):
        num_tokens = (acquire * self.tokenlen) / 100
        self.get_mnlp(data, model, num_tokens)

    def get_mnlp(self, dataset, model, num_tokens, batch_size=50):
        for layer in model.layers:
            layer.trainable = False
        tm = time.time()
        probs = np.ones(len(dataset)) * float('Inf')
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]

        data_batches = self.create_batches(new_dataset, batch_size=batch_size, str_words=True, tag_padded=False)
        probscores = []
        for data in data_batches:

            words = data['words']
            chars = data['chars']
            caps = data['caps']
            mask = data['tagsmask']

            words = Variable(torch.LongTensor(words))
            chars = Variable(torch.LongTensor(chars))
            caps = Variable(torch.LongTensor(caps))
            mask = Variable(torch.LongTensor(mask))

            wordslen = data['wordslen']
            charslen = data['charslen']



            score = model.get_marginal_prob(x,mask)

            norm_scores = score / np.array(wordslen)
            assert len(norm_scores) == len(words)
            probscores.extend(list(norm_scores[np.array(key=lambda p: len(p[0]), reverse=True)]))

        assert len(new_datapoints) == len(probscores)
        probs[new_datapoints] = np.array(probscores)

        test_indices = np.argsort(probs)
        cur_tokens = 0
        cur_indices = set()
        i = 0
        while cur_tokens < num_tokens:
            cur_indices.add(test_indices[i])
            cur_tokens += len(dataset[test_indices[i]]['words'])
            i += 1
        self.train_index.update(cur_indices)

        print ('D Acquisition took %d seconds:' % (time.time() - tm))


