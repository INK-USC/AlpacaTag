import numpy as np
import time
from scipy import stats
from utils import *
from active.layers import CRF

class Acquisition(object):

    #currently train_data = [['EU','rejects','it',,,,],]
    def __init__(self, train_data, init_percent=2, seed=0):
        self.tokenlen = sum([len(x) for x in train_data])
        self.train_index = set()
        self.obtain_data(train_data, acquire=init_percent)

    def get_mnlp(self, dataset, model, decoder, num_tokens, batch_size=50):
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

            score = CRF.get_marginal_prob(x,mask)

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


    def obtain_data(self, data, model, decoder, acquire=2, num_samples=100):

        num_tokens = (acquire * self.tokenlen) / 100
        self.get_mnlp(data, model, decoder, num_tokens)

    def create_batches(self, dataset, batch_size, order='keep', str_words=False, tag_padded= True):

        newdata = copy.deepcopy(dataset)
        if order=='sort':
            newdata.sort(key = lambda x:len(x['words']))
        elif order=='random':
            random.shuffle(newdata)

        newdata = np.array(newdata)
        batches = []
        num_batches = np.ceil(len(dataset)/float(batch_size)).astype('int')

        for i in range(num_batches):
            batch_data = newdata[(i*batch_size):min(len(dataset),(i+1)*batch_size)]

            words_seqs = [itm['words'] for itm in batch_data]
            caps_seqs = [itm['caps'] for itm in batch_data]
            target_seqs = [itm['tags'] for itm in batch_data]
            chars_seqs = [itm['chars'] for itm in batch_data]
            str_words_seqs = [itm['str_words'] for itm in batch_data]

            seq_pairs = sorted(zip(words_seqs, caps_seqs, target_seqs, chars_seqs, str_words_seqs,
                                   range(len(words_seqs))), key=lambda p: len(p[0]), reverse=True)

            words_seqs, caps_seqs, target_seqs, chars_seqs, str_words_seqs, sort_info = zip(*seq_pairs)
            words_lengths = np.array([len(s) for s in words_seqs])

            words_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in words_seqs])
            caps_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in caps_seqs])

            if tag_padded:
                target_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in target_seqs])
            else:
                target_padded = target_seqs

            words_mask = (words_padded!=0).astype('int')

            chars_pseqs = [pad_seq(s, max(words_lengths), []) for s in chars_seqs]
            chars_lengths = np.array([[len(s) for s in w] for w in chars_pseqs]).reshape(-1)
            chars_padded = np.array([[pad_seq(s, np.max(chars_lengths))
                                      for s in w] for w in chars_pseqs]).reshape(-1,np.max(chars_lengths))

            if str_words:
                outputdict = {'words':words_padded, 'caps':caps_padded, 'tags': target_padded,
                              'chars': chars_padded, 'wordslen': words_lengths, 'charslen': chars_lengths,
                              'tagsmask':words_mask, 'str_words': str_words_seqs, 'sort_info': sort_info}
            else:
                outputdict = {'words':words_padded, 'caps':caps_padded, 'tags': target_padded,
                              'chars': chars_padded, 'wordslen': words_lengths, 'charslen': chars_lengths,
                              'tagsmask':words_mask, 'sort_info': sort_info}

            batches.append(outputdict)

            return batches