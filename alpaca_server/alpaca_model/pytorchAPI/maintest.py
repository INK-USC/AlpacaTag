from .preprocessing import IndexTransformer
from .trainer import Trainer
from .utils import filter_embeddings, load_data_and_labels, load_glove, prepare_dataset
from .models import CNN_BiLSTM_CRF
import torch

p = IndexTransformer(initial_vocab=None, use_char=True)
x_train, y_train = load_data_and_labels('../train.bio')
sent = x_train
label = y_train
p.word_fit(sent)
p.label_fit(label)

dataset = prepare_dataset(sent, label, p)

word_vocab_size = len(p._word_vocab.vocab)
word_embedding_dim = 100
word_hidden_dim = 200
char_vocab_size = len(p._char_vocab.vocab)
char_embedding_dim = 25
char_out_channels = 25

embeddings = load_glove('../glove.6B.100d.txt')
embeddings = filter_embeddings(embeddings, p._word_vocab.vocab, word_embedding_dim)

model = CNN_BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                       char_embedding_dim, char_out_channels, p._label_vocab.vocab , pretrained=embeddings)

learning_rate = 0.01
print('Initial learning rate is: %s' % (learning_rate))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

trainer = Trainer(model, optimizer)
trainmodel = trainer.train_model(3, dataset,learning_rate=learning_rate, batch_size=10,lr_decay=0.05)

