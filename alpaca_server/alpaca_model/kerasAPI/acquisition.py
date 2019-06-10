import time


class Acquisition(object):
    #currently train_data = [['EU','rejects','it',,,,],]
    def __init__(self, train_data, model, preprocess, init_percent=2, seed=0):
        self.tokenlen = sum([len(x) for x in train_data])
        self.train_index = set()
        self.return_index = set()
        self.obtain_data(train_data, model, preprocess, acquire=init_percent)

    def obtain_data(self, x_train, model, preprocess, acquire=2):
        num_tokens = (acquire * self.tokenlen) / 100
        return self.get_mnlp(x_train, model, preprocess, num_tokens)

    def get_mnlp(self, x_train, model, preprocess, num_tokens, batch_size=50):
        tm = time.time()
        probs = np.ones(len(x_train)) * float('Inf')
        new_dataset = [datapoint for j, datapoint in enumerate(x_train)]
        new_datapoints = [j for j in range(len(x_train)) if j not in self.train_index]

        idx_max = math.ceil(len(new_datapoints) / batch_size)
        probscores = []
        for idx in range(idx_max):
            batch_x = [new_dataset[i] for i in new_datapoints[idx * batch_size: (idx + 1) * batch_size]]
            wordslen = [len(x) for x in batch_x]
            wordslen = np.array(wordslen)

            lengths = map(len, batch_x)
            pdata = preprocess.transform(batch_x)
            score = model.predict(pdata)
            score = [iy[:l] for iy, l in zip(score, lengths)]
            score = np.array(score)
            scores = [x[-1][-1] for x in score]
            scores = np.array(scores)
            norm_scores = scores/wordslen
            assert len(norm_scores) == len(batch_x)
            probscores.extend(list(norm_scores[wordslen.argsort()[::-1]]))

        assert len(new_datapoints) == len(probscores)
        probs[new_datapoints] = np.array(probscores)

        test_indices = np.argsort(probs)
        cur_tokens = 0
        cur_indices = set()
        i = 0
        while len(cur_indices) < 100:  #num_token
            cur_indices.add(test_indices[i])
            cur_tokens += len(x_train[test_indices[i]])
            i += 1
        self.return_index = set()
        self.return_index.update(cur_indices)
        self.train_index.update(cur_indices)
        print ('D Acquisition took %d seconds:' % (time.time() - tm))
        return cur_indices

