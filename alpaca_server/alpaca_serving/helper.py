import argparse
import logging
import os
import sys
import time
import uuid
import warnings

import zmq
from termcolor import colored
from zmq.utils import jsonapi



def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


def send_ndarray(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    return src.send_multipart([dest, jsonapi.dumps(md), X, req_id], flags, copy=copy, track=track)

def send_test(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    return src.send_multipart([dest, X, req_id], flags, copy=copy, track=track)

def check_max_seq_len(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        ivalue = int(value)
        if ivalue <= 3:
            raise argparse.ArgumentTypeError("%s is an invalid int value must be >3 "
                                             "(account for maximum three special symbols in alpaca_model) or NONE" % value)
    except TypeError:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def get_args_parser():
    from . import __version__

    parser = argparse.ArgumentParser(description='Start a AlpacaServer for alpaca_server')

    group1 = parser.add_argument_group('File Paths',
                                       'config the path, checkpoint and filename of a pretrained/fine-tuned alpaca_model')
    # group1.add_argument('-model_dir', type=str, required=True,
    #                     help='directory of a pretrained alpaca_model')
    group1.add_argument('-model_dir', type=str,
                        help='directory of a pretrained alpaca_model')

    group2 = parser.add_argument_group('Parameters',
                                       'config how alpaca_model and pooling works')
    group2.add_argument('-max_seq_len', type=check_max_seq_len, default=25,
                        help='maximum length of a sequence, longer sequence will be trimmed on the right side. '
                             'set it to NONE for dynamically using the longest sequence in a (mini)batch.')
    group3 = parser.add_argument_group('Serving Configs',
                                       'config how server utilizes GPU/CPU resources')
    group3.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    group3.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')
    group3.add_argument('-http_port', type=int, default=None,
                        help='server port for receiving HTTP requests')
    group3.add_argument('-http_max_connect', type=int, default=10,
                        help='maximum number of concurrent HTTP connections')
    group3.add_argument('-cors', type=str, default='*',
                        help='setting "Access-Control-Allow-Origin" for HTTP requests')
    group3.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    group3.add_argument('-batch_size', type=int, default=256,
                        help='batch size')
    group3.add_argument('-epoch', type=int, default=256,
                        help='epoch')
    group3.add_argument('-cpu', action='store_true', default=False,
                        help='running on CPU (default on GPU)')
    group3.add_argument('-xla', action='store_true', default=False,
                        help='enable XLA compiler (experimental)')
    group3.add_argument('-fp16', action='store_true', default=False,
                        help='use float16 precision (experimental)')
    group3.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determine the fraction of the overall amount of memory \
                        that each visible GPU should be allocated per worker. \
                        Should be in range [0.0, 1.0]')
    group3.add_argument('-device_map', type=int, nargs='+', default=[],
                        help='specify the list of GPU device ids that will be used (id starts from 0). \
                        If num_worker > len(device_map), then device will be reused; \
                        if num_worker < len(device_map), then device_map[:num_worker] will be used')
    group3.add_argument('-prefetch_size', type=int, default=10,
                        help='the number of batches to prefetch on each worker. When running on a CPU-only machine, \
                        this is set to 0 for comparability')
    group3.add_argument('-fixed_embed_length', action='store_true', default=False,
                        help='when "max_seq_len" is set to None, the server determines the "max_seq_len" according to '
                             'the actual sequence lengths within each batch. When "pooling_strategy=NONE", '
                             'this may cause two ".encode()" from the same client results in different sizes [B, T, D].'
                             'Turn this on to fix the "T" in [B, T, D] to "max_position_embeddings" in  json config.')

    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser

def auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://127.0.0.1')
    else:
        # Get the location for tmp file for sockets
        try:
            tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
            if not os.path.exists(tmp_dir):
                raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
            tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
        except KeyError:
            tmp_dir = '*'

        socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')


def get_run_args(parser_fn=get_args_parser, printed=True):
    args = parser_fn().parse_args()
    if printed:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


def get_benchmark_parser():
    parser = get_args_parser()
    parser.description = 'Benchmark AlpacaServer locally'

    parser.set_defaults(num_client=1, client_batch_size=4096)

    group = parser.add_argument_group('Benchmark parameters', 'config the experiments of the benchmark')

    group.add_argument('-test_client_batch_size', type=int, nargs='*', default=[1, 16, 256, 4096])
    group.add_argument('-test_max_batch_size', type=int, nargs='*', default=[8, 32, 128, 512])
    group.add_argument('-test_max_seq_len', type=int, nargs='*', default=[32, 64, 128, 256])
    group.add_argument('-test_num_client', type=int, nargs='*', default=[1, 4, 16, 64])
    group.add_argument('-test_pooling_layer', type=int, nargs='*', default=[[-j] for j in range(1, 13)])

    group.add_argument('-wait_till_ready', type=int, default=30,
                       help='seconds to wait until server is ready to serve')
    group.add_argument('-client_vocab_file', type=str, default='README.md',
                       help='file path for building client vocabulary')
    group.add_argument('-num_repeat', type=int, default=10,
                       help='number of repeats per experiment (must >2), '
                            'as the first two results are omitted for warm-up effect')
    return parser


def get_shutdown_parser():
    parser = argparse.ArgumentParser()
    parser.description = 'Shutting down a AlpacaServer instance running on a specific port'

    parser.add_argument('-ip', type=str, default='localhost',
                        help='the ip address that a AlpacaServer is running on')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, required=True,
                        help='the port that a AlpacaServer is running on')
    parser.add_argument('-timeout', type=int, default=1000,
                        help='timeout (ms) for connecting to a server')
    return parser


class TimeContext:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        self.start = time.perf_counter()
        print(self._msg, end=' ...\t', flush=True)

    def __exit__(self, typ, value, traceback):
        self.duration = time.perf_counter() - self.start
        print(colored('    [%3.3f secs]' % self.duration, 'green'), flush=True)
