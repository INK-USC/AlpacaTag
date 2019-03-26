import torch
import torch.nn as nn
import math

from torch.autograd import Variable
import neural_ner
from neural_ner.util.utils import *
from torch.nn.utils.rnn import PackedSequence
from torch.nn.parameter import Parameter
from torch.nn import _VF
from torch._jit_internal import weak_module, weak_script_method, weak_script

_rnn_impls = {
    'GRU': _VF.gru,
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
    'LSTM': _VF.lstm,
}

@weak_script
def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)

class RNNBase_BB(nn.Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', '_flat_parameters']

    def __init__(self, mode, input_size, hidden_size, sigma_prior,
                 num_layers=1, batch_first=False,
                 dropout=0, bidirectional=True):

        super(RNNBase_BB, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.sampled_weights = []
        self.sigma_prior = sigma_prior

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self.means = []
        self.logvars = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih_mu = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh_mu = Parameter(torch.Tensor(gate_size, hidden_size))
                w_ih_logvar = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh_logvar = Parameter(torch.Tensor(gate_size, hidden_size))

                b_ih_mu = Parameter(torch.Tensor(gate_size))
                b_hh_mu = Parameter(torch.Tensor(gate_size))
                b_ih_logvar = Parameter(torch.Tensor(gate_size))
                b_hh_logvar = Parameter(torch.Tensor(gate_size))

                self.means += [w_ih_mu, w_hh_mu, b_ih_mu, b_hh_mu]
                self.logvars += [w_ih_logvar, w_hh_logvar, b_ih_logvar, b_hh_logvar]

                layer_params = (w_ih_mu, w_ih_logvar, w_hh_mu, w_hh_logvar, b_ih_mu, b_ih_logvar, b_hh_mu, b_hh_logvar)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l_mu{}{}', 'weight_ih_l_logvar{}{}', 'weight_hh_l_mu{}{}',
                               'weight_hh_l_logvar{}{}']
                param_names += ['bias_ih_l_mu{}{}', 'bias_ih_l_logvar{}{}', 'bias_hh_l_mu{}{}', 'bias_hh_l_logvar{}{}']

                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()
        self.lpw = 0
        self.lqw = 0

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (8),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(RNNBase_BB, self)._apply(fn)
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        logvar_init = math.log(stdv) * 2
        for mean in self.means:
            mean.data.uniform_(-stdv, stdv)
        for logvar in self.logvars:
            logvar.data.fill_(logvar_init)

    def get_flat_weights(self):
        return self._flat_weights

    @weak_script_method
    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    @weak_script_method
    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    @weak_script_method
    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def sample(self, usecuda=True):
        self.sampled_weights = []
        for i in range(len(self.means)):
            mean = self.means[i]
            logvar = self.logvars[i]
            eps = torch.zeros(mean.size())
            if usecuda:
                eps = eps.cuda()

            eps.normal_(0, self.sigma_prior)
            std = logvar.mul(0.5).exp()
            weight = mean + Variable(eps) * std
            self.sampled_weights.append(weight)

    def _calculate_prior(self, weights):
        lpw = 0.
        for w in weights:
            lpw += log_gaussian(w, 0, self.sigma_prior).sum()
        return lpw

    def _calculate_posterior(self, weights):
        lqw = 0.
        for i, w in enumerate(weights):
            lqw += log_gaussian_logsigma(w, self.means[i], 0.5 * self.logvars[i]).sum()
        return lqw

    def forward(self, input, hx=None, usecuda=True):
        if self.training:
            self.sample(usecuda=usecuda)
            weights = self.sampled_weights
            self.lpw = self._calculate_prior(weights)
            self.lqw = self._calculate_posterior(weights)
        else:
            weights = self.means

        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,max_batch_size,
                             self.hidden_size,dtype=input.dtype, device=input.device)
            if self.mode == 'LSTM':
                hx = (hx, hx)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls[self.mode]
        if batch_sizes is None:
            result = _impl(input, hx, self.get_flat_weights(), self.bias, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self.get_flat_weights(), self.bias,
                           self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(RNNBase_BB, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l_mu{}{}', 'weight_ih_l_logvar{}{}', 'weight_hh_l_mu{}{}',
                               'weight_hh_l_logvar{}{}']
                param_names += ['bias_ih_l_mu{}{}', 'bias_ih_l_logvar{}{}', 'bias_hh_l_mu{}{}', 'bias_hh_l_logvar{}{}']

                param_names = [x.format(layer, suffix) for x in param_names]
                self._all_weights += [param_names]

    @property
    def _flat_weights(self):
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

class LSTM_BB(RNNBase_BB):

    def __init__(self, *args, **kwargs):
        super(LSTM_BB, self).__init__('LSTM', *args, **kwargs)

class baseRNN_BB(nn.Module):

    def __init__(self, vocab_size, hidden_size, input_dropout_p, output_dropout_p, n_layers, rnn_cell,
                 max_len=25):

        super(baseRNN_BB, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len

        self.input_dropout_p = input_dropout_p
        self.output_dropout_p = output_dropout_p

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = LSTM_BB
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.input_dropout = nn.Dropout(p=input_dropout_p)


class WordEncoderRNN_BB(baseRNN_BB):

    def __init__(self, vocab_size, embedding_size, hidden_size, char_size, cap_size, sigma_prior, input_dropout_p=0.5,
                 output_dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='lstm'):

        super(WordEncoderRNN_BB, self).__init__(vocab_size, hidden_size, input_dropout_p,
                                                output_dropout_p, n_layers, rnn_cell)

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        augmented_embedding_size = embedding_size + char_size + cap_size
        self.rnn = self.rnn_cell(augmented_embedding_size, hidden_size, n_layers,
                                 bidirectional=bidirectional, dropout=output_dropout_p,
                                 batch_first=True)

    def forward(self, words, char_embedding, cap_embedding, input_lengths):

        embedded = self.embedding(words)
        if cap_embedding is not None:
            embedded = torch.cat((embedded, char_embedding, cap_embedding), 2)
        else:
            embedded = torch.cat((embedded, char_embedding), 2)

        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, _ = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output

    def get_lpw_lqw(self):

        lpw = self.rnn.lpw
        lqw = self.rnn.lqw
        return lpw, lqw