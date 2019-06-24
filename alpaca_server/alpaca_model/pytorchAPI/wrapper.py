from .active_learning import *
from .preprocessing import *
from .trainer import *
from .utils import *
from .models import *
from .tagger import *
import torch

class SequenceTaggingModel(object):

    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True,
                 initial_vocab=None,
                 optimizer='adam'):

        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.embeddings = load_glove('glove.6B.100d.txt')
        self.use_char = use_char
        self.use_crf = use_crf
        self.initial_vocab = initial_vocab
        self.optimizer = optimizer

        self.trainer = None
        self.activemodel = None
        self.model = None
        self.p = None
        self.tagger = None
        self.loss = None
        self.labeled = set()
        self.acquisition = None

        # model name like CNN_BiLSTM_CRF
        self.model_name = None

    def analyze(self, text, tokenizer=str.split):
        """Analyze text and return pretty format.

        Args:
            text: string, the input text.
            tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.

        Returns:
            res: dict.
        """
        if not self.tagger:
            self.tagger = Tagger(self.model, preprocessor=self.p, tokenizer=tokenizer)

        return self.tagger.analyze(text)

    def save(self, model_name):
        p_path = model_name + '.pre'
        m_path = model_name + '.pt'
        self.p.save(p_path)
        torch.save(self.model.state_dict(), m_path)

    def load(self, model_name):
        p_path = model_name + '.pre'
        m_path = model_name + '.pt'
        self.p = IndexTransformer.load(p_path)
        embeddings = filter_embeddings(self.embeddings, self.p._word_vocab.vocab, self.word_embedding_dim)
        self.model = CNN_BiLSTM_CRF(self.p.word_vocab_size,
                                    self.word_embedding_dim,
                                    self.word_lstm_size,
                                    self.p.char_vocab_size,
                                    self.char_embedding_dim,
                                    self.char_lstm_size,
                                    self.p._label_vocab.vocab, pretrained=embeddings)
        self.model.load_state_dict(torch.load(m_path))

    # list of sentences into train_sentences [['EU rejects ~~'],[''],..]
    def online_word_build(self, x_train, predefined_label):
        # pretrained_label = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        # let's make it to text file and add it altogether!
        # x_train = []
        # for sentence in train_sentences:
        #     x_train.append(str.split(sentence))
        self.p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        self.p.label_fit(predefined_label)
        self.p.word_fit(x_train)

        embeddings = filter_embeddings(self.embeddings, self.p._word_vocab.vocab, self.word_embedding_dim)
        self.model = CNN_BiLSTM_CRF(self.p.word_vocab_size,
                                    self.word_embedding_dim,
                                    self.word_lstm_size,
                                    self.p.char_vocab_size,
                                    self.char_embedding_dim,
                                    self.char_lstm_size,
                                    self.p._label_vocab.vocab, pretrained=embeddings)

        learning_rate = 0.01
        print('Initial learning rate is: %s' % (learning_rate))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.trainer = Trainer(self.model, optimizer)

        return self.p

    def active_learning(self, x_train, y_train, acquire_method_name):

        if self.acquisition is None:
            self.acquisition = Acquisition(x_train, init_percent=self.initial_vocab,
                                           seed=0, acq_mode = self.model)

        self.acquisition.obtain_data(data=x_train, model=self.model, model_name=self.model_name, method=acquire_method_name)

        return [i for i in self.acquisition.train_index]


    def online_learning(self, x_train, y_train, epochs=5, batch_size=5, verbose=1, callbacks=None, shuffle=True):

        dataset = prepare_dataset(x_train, y_train, self.p)
        learning_rate = 0.01

        self.model = self.trainer.train_model(num_epochs=3, train_data=dataset, learning_rate=learning_rate, batch_size=10, lr_decay=0.05)


    def noun_chunks(self, text):
        nlp = spacy.load('en_core_web_sm')
        chunk_merger = NounChunk(nlp)
        doc = chunk_merger.chunking(text)
        return doc


