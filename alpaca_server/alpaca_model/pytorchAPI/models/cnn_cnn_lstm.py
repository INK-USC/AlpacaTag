import torch
import torch.nn as nn

from alpaca_model.pytorchAPI.initializer import Initializer
from alpaca_model.pytorchAPI.modules import CharEncoderCNN
from alpaca_model.pytorchAPI.modules import WordEncoderCNN
from alpaca_model.pytorchAPI.modules import DecoderRNN


class CNN_CNN_LSTM(nn.Module):

    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                 char_embedding_dim, char_out_channels, decoder_hidden_units, tag_to_id, cap_input_dim=4,
                 cap_embedding_dim=0, pretrained=None):

        super(CNN_CNN_LSTM, self).__init__()

        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels

        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_out_channels = char_out_channels

        self.cap_input_dim = cap_input_dim
        self.cap_embedding_dim = cap_embedding_dim

        self.tag_to_ix = tag_to_id
        self.tagset_size = len(tag_to_id)

        self.initializer = Initializer()

        if self.cap_input_dim and self.cap_embedding_dim:
            self.cap_embedder = nn.Embedding(self.cap_input_dim, self.cap_embedding_dim)
            self.initializer.init_embedding(self.cap_embedder.weight)

        self.char_encoder = CharEncoderCNN(char_vocab_size, char_embedding_dim, char_out_channels,
                                           kernel_width=3, pad_width=1)

        self.initializer.init_embedding(self.char_encoder.embedding.weight)

        self.word_encoder = WordEncoderCNN(word_vocab_size, word_embedding_dim, char_out_channels,
                                           kernel_width = 5, pad_width = 2, input_dropout_p=0.5,
                                           output_dropout_p=0.5, out_channels=word_out_channels)

        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        augmented_decoder_inp_size = (word_out_channels + word_embedding_dim +
                                      char_out_channels + cap_embedding_dim)
        self.decoder = DecoderRNN(augmented_decoder_inp_size, decoder_hidden_units, self.tagset_size,
                                  self.tag_to_ix, input_dropout_p=0.5)

    def forward(self, words, tags, chars, caps, wordslen, charslen, tagsmask, n_batches, usecuda=True):

        batch_size, max_len = words.size()

        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None

        char_features = self.char_encoder(chars)
        char_features = char_features.view(batch_size, max_len, -1)

        word_features, word_input_feats = self.word_encoder(words, char_features, cap_features)

        new_word_features = torch.cat((word_features,word_input_feats),2)
        loss = self.decoder(new_word_features, tags, tagsmask, usecuda=usecuda)

        return loss

    def decode(self, words, chars, caps, wordslen, charslen, tagsmask, usecuda=True,
               score_only = False):

        batch_size, max_len = words.size()

        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None

        char_features = self.char_encoder(chars)
        char_features = char_features.view(batch_size, max_len, -1)

        word_features, word_input_feats = self.word_encoder(words, char_features, cap_features)

        new_word_features = torch.cat((word_features,word_input_feats),2)

        if score_only:
            score, _ = self.decoder.decode(new_word_features, tagsmask, wordslen, usecuda=usecuda)
            return score

        score, tag_seq = self.decoder.decode(new_word_features, tagsmask, wordslen, usecuda=usecuda)
        return score, tag_seq
