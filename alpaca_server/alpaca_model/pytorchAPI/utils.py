from collections import Counter
import numpy as np
import copy
import random
import torch
from pathlib import Path
from urllib.parse import urlparse
import os
import requests
import re
import tempfile
from tqdm import tqdm as _tqdm
import logging
import shutil
import gensim
import flair


# from pytorch_pretrained_bert.modeling_openai import (
#     PRETRAINED_MODEL_ARCHIVE_MAP as OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
# )
#
# from pytorch_pretrained_bert import (
#     BertTokenizer,
#     BertModel,
#     TransfoXLTokenizer,
#     TransfoXLModel,
#     OpenAIGPTModel,
#     OpenAIGPTTokenizer,
# )

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


class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {"mininterval": Tqdm.default_mininterval, **kwargs}

        return _tqdm(*args, **new_kwargs)


log = logging.getLogger("AlpacaTag")


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path

    # make HEAD request to check ETag
    response = requests.head(url, headers={"User-Agent": "Flair"})
    if response.status_code != 200:
        raise IOError(
            f"HEAD request failed for url {url} with status code {response.status_code}."
        )

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()
        # logger.info("%s not found in cache, downloading to %s", url, temp_filename)
        log.info("%s not found in cache, downloading to %s", url, temp_filename)

        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "Flair"})
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        with open(temp_filename, "wb") as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        log.info("copying %s to cache at %s", temp_filename, cache_path)
        shutil.copyfile(temp_filename, str(cache_path))
        log.info("removing temp file %s", temp_filename)
        os.close(fd)
        os.remove(temp_filename)

    return cache_path


cache_root = os.path.expanduser(os.path.join("~", ".AlpacaTag"))


def cached_path(url_or_filename: str, cache_dir: Path) -> Path:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    dataset_cache = Path(cache_root) / cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and Path(url_or_filename).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def get_Distilbert_embeddings(vocab, dim):
    from sentence_tranformers import SentenceTransformer

    sentence_transformer = SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")
    _embeddings = np.array(sentence_transformer.encode(list(vocab.keys()), show_progress_bar=False))
    assert _embeddings.shape[1] == dim, f"Bad word_embedding_dim set: {dim} (found {_embeddings.shape[1]})"
    return _embeddings


def get_Bert_embeddings(vocab, dim):
    from flair.embeddings import BertEmbeddings
    from flair.data import Sentence

    _embeddings = np.zeros([len(vocab), dim])
    temp = []
    for each_word in vocab:
        temp.append(each_word)
    sentence = Sentence(' '.join(temp))

    embedding = BertEmbeddings()

    embedding.embed(sentence)
    for token in sentence:
        try:
            _embeddings[vocab[token.text]] = token.embedding
        except KeyError:
            log.warning(f'Bad token {token.text} for Bert embedding')

    return _embeddings


def get_GPT_embeddings(vocab, dim):
    _embeddings = np.zeros([len(vocab), dim])

    if "openai-gpt" not in OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP.keys():
        raise ValueError("Provided OpenAI GPT model is not available.")
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    gpt_model = OpenAIGPTModel.from_pretrained("openai-gpt")

    with torch.no_grad():
        for word in vocab:
            subwords = tokenizer.tokenize(word)
            indexed_tokens = tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(flair.device)
            hidden_states = gpt_model(tokens_tensor)

            first_embedding = hidden_states[0][0]
            last_embedding = hidden_states[0][len(hidden_states[0]) - 1]
            final_embedding = torch.cat([first_embedding, last_embedding])

            _embeddings[vocab[word]] = final_embedding

    return _embeddings


def get_Elmo_embeddings(vocab, dim):

    _embeddings = np.zeros([len(vocab), dim])

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

    from flair import device
    import allennlp.commands.elmo

    if re.fullmatch(r"cuda:[0-9]+", str(device)):
        cuda_device = int(str(device).split(":")[-1])
    elif str(device) == "cpu":
        cuda_device = -1
    else:
        cuda_device = 0

    elmo_embeddings = allennlp.commands.elmo.ElmoEmbedder(
        options_file=options_file, weight_file=weight_file, cuda_device=cuda_device
    )

    temp = []
    for each_word in vocab:
        temp.append(each_word)
    sentences_words = [temp]

    embeddings = elmo_embeddings.embed_batch(sentences_words)
    sentence_embeddings = embeddings[0]
    for token, token_idx in zip(sentences_words[0], range(len(sentences_words[0]))):

        word_embedding = torch.cat(
            [
                torch.FloatTensor(sentence_embeddings[0, token_idx, :]),
                torch.FloatTensor(sentence_embeddings[1, token_idx, :]),
                torch.FloatTensor(sentence_embeddings[2, token_idx, :]),
            ],
            0,
        )

        _embeddings[vocab[token]] = word_embedding

    return _embeddings


def get_glove_embeddings(vocab, dim):

    old_base_path = ("https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/")
    cache_dir = Path("embeddings")
    cached_path(f"{old_base_path}glove.gensim.vectors.npy", cache_dir=cache_dir)
    embeddings_path = cached_path(f"{old_base_path}glove.gensim", cache_dir=cache_dir)
    precomputed_word_embeddings = gensim.models.KeyedVectors.load(
        str(embeddings_path)
    )

    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in precomputed_word_embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = precomputed_word_embeddings[word]

    return _embeddings


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
