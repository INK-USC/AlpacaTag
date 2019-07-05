from collections import Counter
import numpy as np
import copy
import random
import torch

START_TAG = '<START>'
STOP_TAG = '<STOP>'

def load_data_and_labels(filename, encoding='utf-8'):
    sents, labels = [], []
    words, tags = [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split(' ')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []

    return sents, labels


def load_sentences(sentences):
    sents = []
    for sent in sentences:
        sents.append(sent.split())
    return sents


def filter_embeddings(embeddings, vocab, dim):
    """Loads word vectors in numpy array.

    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.

    Returns:
        numpy array: an array of word embeddings.
    """
    if not isinstance(embeddings, dict):
        return
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word]

    return _embeddings


def load_glove(file):
    """Loads GloVe vectors in numpy array.

    Args:
        file (str): a path to a glove file.

    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(file, encoding="utf8", errors='ignore') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def pad_seq(seq, max_length, PAD_token=0):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def prepare_dataset(sentences, tags, p, lower=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    word_to_id = p._word_vocab.vocab
    char_to_id = p._char_vocab.vocab
    tag_to_id = p._label_vocab.vocab

    def f(x): return x.lower() if lower else x
    data = []
    if tags is None:
        for s in sentences:
            str_words = [w for w in s]
            words = [word_to_id[f(w) if f(w) in word_to_id else '<unk>']
                     for w in str_words]
            # Skip characters that are not in the training set
            chars = [[char_to_id[c] for c in w if c in char_to_id]
                     for w in str_words]
            caps = [cap_feature(w) for w in str_words]
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'caps': caps,
                'tags': None,
            })
    else:
        for s,ts in zip(sentences,tags):
            str_words = [w for w in s]
            words = [word_to_id[f(w) if f(w) in word_to_id else '<unk>']
                     for w in str_words]
            # Skip characters that are not in the training set
            chars = [[char_to_id[c] for c in w if c in char_to_id]
                     for w in str_words]
            caps = [cap_feature(w) for w in str_words]
            tags = [tag_to_id[t] for t in ts]
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'caps': caps,
                'tags': tags,
            })
    return data


def create_batches(dataset, batch_size, order='keep', str_words=False, tag_padded= True):

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


def log_sum_exp(vec, dim=-1, keepdim = False):
    max_score, _ = vec.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = vec - max_score
    else:
        stable_vec = vec - max_score.unsqueeze(dim)
    output = max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()
    return output


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


def bayes_loss_function(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)


def get_entities(seq, suffix=False):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start