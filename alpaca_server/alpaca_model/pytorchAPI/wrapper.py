from alpaca_model.pytorchAPI.active_learning import Acquisition
from alpaca_model.pytorchAPI.preprocessing import IndexTransformer
from alpaca_model.pytorchAPI.trainer import Trainer
from alpaca_model.pytorchAPI.models import CNN_BiLSTM_CRF
from alpaca_model.pytorchAPI.tagger import Tagger
from alpaca_model.pytorchAPI.utils import get_Elmo_embeddings, prepare_dataset
import torch

GLOVE_EMBEDDING_SIZE = 100
ELMO_EMBEDDING_SIZE = 768
GPT_EMBEDDING_SIZE = 1536
BERT_EMBEDDING_SIZE = 3072

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
        self.embedding = embeddings
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

        # embeddings = get_Elmo_embeddings(self.p._word_vocab.vocab, ELMO_EMBEDDING_SIZE)
        # embeddings = get_glove_embeddings(self.p._word_vocab.vocab, self.word_embedding_dim)
        # embeddings = get_GPT_embeddings(self.p._word_vocab.vocab, GPT_EMBEDDING_SIZE)
        # embeddings = filter_embeddings(self.embeddings, self.p._word_vocab.vocab, self.word_embedding_dim)
        embeddings = get_Bert_embeddings()(self.p._word_vocab.vocab, BERT_EMBEDDING_SIZE)
        self.model = CNN_BiLSTM_CRF(self.p.word_vocab_size,
                                    self.word_embedding_dim,
                                    self.word_lstm_size,
                                    self.p.char_vocab_size,
                                    self.char_embedding_dim,
                                    self.char_lstm_size,
                                    self.p._label_vocab.vocab, pretrained=embeddings)

        self.model.load_state_dict(torch.load(m_path))
        learning_rate = 0.01
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.trainer = Trainer(self.model, optimizer)

    # initiate - if setting change -> must execute initiate
    # list of sentences into train_sentences [['EU rejects ~~'],[''],..]
    def online_word_build(self, x_train, predefined_label):
        self.p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        self.p.label_fit(predefined_label)
        self.p.word_fit(x_train)

        embeddings = get_Elmo_embeddings(self.p._word_vocab.vocab, ELMO_EMBEDDING_SIZE)
        # embeddings = get_glove_embeddings(self.p._word_vocab.vocab, self.word_embedding_dim)
        # embeddings = filter_embeddings(self.embeddings, self.p._word_vocab.vocab, self.word_embedding_dim)
        # embeddings = get_GPT_embeddings(self.p._word_vocab.vocab, GPT_EMBEDDING_SIZE)
        # embeddings = get_Bert_embeddings(self.p._word_vocab.vocab, BERT_EMBEDDING_SIZE)
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

    def active_learning(self, x_train, size):
        dataset = prepare_dataset(x_train, None, self.p)
        if self.acquisition is None:
            self.acquisition = Acquisition(dataset, size=size, seed=0, acq_mode='d')

        self.acquisition.obtain_data(data=dataset, model=self.model, acquire=size, method='mnlp')
        print(self.acquisition.return_index)
        return self.acquisition.return_index, self.acquisition.return_score

    def online_learning(self, x_train, y_train, epochs=5, batch_size=5):
        dataset = prepare_dataset(x_train, y_train, self.p)
        learning_rate = 0.01

        self.model = self.trainer.train_model(num_epochs=epochs, train_data=dataset, learning_rate=learning_rate, batch_size=batch_size, lr_decay=0.05)
