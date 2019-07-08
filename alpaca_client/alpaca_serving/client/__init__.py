import sys
import threading
import time
import uuid
from collections import namedtuple
from functools import wraps

import numpy as np
import zmq
from zmq.utils import jsonapi

__all__ = ['__version__', 'AlpacaClient', 'ConcurrentAlpacaClient']

# in the future client version must match with server version
__version__ = '1.0.1'
if sys.version_info >= (3, 0):
    from ._py3_ import *
else:
    from ._py2_ import *

_Response = namedtuple('_Response', ['id', 'content'])
Response = namedtuple('Response', ['id'])

class AlpacaClient(object):
    def __init__(self, ip='localhost', port=5555, port_out=5556,
                 output_fmt='ndarray', show_server_config=False,
                 identity=None, check_version=True, check_length=True,
                 check_token_info=True, ignore_all_checks=False,
                 timeout=15000):

        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.identity = identity or str(uuid.uuid4()).encode('ascii')
        self.sender.connect('tcp://%s:%d' % (ip, port))

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)
        self.receiver.connect('tcp://%s:%d' % (ip, port_out))

        self.request_id = 0
        self.timeout = timeout
        self.pending_request = set()
        self.pending_response = {}

        self.output_fmt = output_fmt
        self.port = port
        self.port_out = port_out
        self.ip = ip
        self.length_limit = 0
        self.token_info_available = False

        if not ignore_all_checks and (check_version or show_server_config or check_length or check_token_info):
            s_status = self.server_status

            if check_version and s_status['server_version'] != self.status['client_version']:
                raise AttributeError('version mismatch! server version is %s but client version is %s!\n'
                                     'consider "pip install -U bert-serving-server bert-serving-client"\n'
                                     'or disable version-check by "BertClient(check_version=False)"' % (
                                         s_status['server_version'], self.status['client_version']))

            if check_length:
                if s_status['max_seq_len'] is not None:
                    self.length_limit = int(s_status['max_seq_len'])
                else:
                    self.length_limit = None

            if check_token_info:
                self.token_info_available = bool(s_status['show_tokens_to_client'])

            if show_server_config:
                self._print_dict(s_status, 'server config:')

    def close(self):
        """
            Gently close all connections of the client. If you are using BertClient as context manager,
            then this is not necessary.
        """
        self.sender.close()
        self.receiver.close()
        self.context.term()

    def _send(self, msg_type, msg, msg_len=0):
        self.request_id += 1
        self.sender.send_multipart([self.identity, msg_type, msg, b'%d' % self.request_id, b'%d' % msg_len])
        self.pending_request.add(self.request_id)
        return self.request_id

    def _recv(self, wait_for_req_id=None):
        try:
            while True:
                # a request has been returned and found in pending_response
                if wait_for_req_id in self.pending_response:
                    response = self.pending_response.pop(wait_for_req_id)
                    return _Response(wait_for_req_id, response)

                # receive a response
                response = self.receiver.recv_multipart()
                request_id = int(response[-1])
                # if not wait for particular response then simply return
                if not wait_for_req_id or (wait_for_req_id == request_id):
                    self.pending_request.remove(request_id)
                    return _Response(request_id, response)
                elif wait_for_req_id != request_id:
                    self.pending_response[request_id] = response
                    # wait for the next response
        except Exception as e:
            raise e
        finally:
            if wait_for_req_id in self.pending_request:
                self.pending_request.remove(wait_for_req_id)

    def _recv_test(self, wait_for_req_id=None):
        request_id, response = self._recv(wait_for_req_id)
        print(request_id)
        print(response)
        return _Response(request_id, response)

    @property
    def status(self):
        """
            Get the status of this BertClient instance
        :rtype: dict[str, str]
        :return: a dictionary contains the status of this BertClient instance
        """
        return {
            'identity': self.identity,
            'num_request': self.request_id,
            'num_pending_request': len(self.pending_request),
            'pending_request': self.pending_request,
            'output_fmt': self.output_fmt,
            'port': self.port,
            'port_out': self.port_out,
            'server_ip': self.ip,
            'client_version': __version__,
            'timeout': self.timeout
        }

    def _timeout(func):
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kwargs)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    'no response from the server (with "timeout"=%d ms), please check the following:'
                    'is the server still online? is the network broken? are "port" and "port_out" correct? '
                    'are you encoding a huge amount of data whereas the timeout is too small for that?' % self.timeout)
                if _py2:
                    raise t_e
                else:
                    _raise(t_e, _e)
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)

        return arg_wrapper

    @property
    @_timeout
    def server_status(self):
        """
            Get the current status of the server connected to this client
        :return: a dictionary contains the current status of the server connected to this client
        :rtype: dict[str, str]
        """
        req_id = self._send(b'SHOW_CONFIG', b'SHOW_CONFIG')
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def initiate(self, project_id):
        # model = Sequence()
        req_id = self._send(b'INITIATE', bytes(str(project_id), encoding='ascii'))
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def online_initiate(self, sentences, predefined_label):
        # model.online_word_build(sent,[['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']])
        req_id = self._send(b'ONLINE_INITIATE', jsonapi.dumps([sentences, predefined_label]), len(sentences))
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def online_learning(self, sentences, labels):
        assert len(sentences) == len(labels)
        req_id = self._send(b'ONLINE_LEARNING', jsonapi.dumps([sentences, labels]), len(sentences))
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def active_learning(self, sentences, num_instances):
        req_id = self._send(b'ACTIVE_LEARNING', jsonapi.dumps([sentences, num_instances]), len(sentences))
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def predict(self, sentences):
        req_id = self._send(b'PREDICT', jsonapi.dumps(sentences), len(sentences))
        return jsonapi.loads(self._recv(req_id).content[1])

    @_timeout
    def encode(self, texts, blocking=True, is_tokenized=False, show_tokens=False):
        req_id = self._send(jsonapi.dumps(texts), len(texts))
        r = self._recv_test(req_id)
        return r

    def fetch(self, delay=.0):
        time.sleep(delay)
        while self.pending_request:
            yield self._recv_ndarray()

    def fetch_all(self, sort=True, concat=False):
        if self.pending_request:
            tmp = list(self.fetch())
            if sort:
                tmp = sorted(tmp, key=lambda v: v.id)
            tmp = [v.embedding for v in tmp]
            if concat:
                if self.output_fmt == 'ndarray':
                    tmp = np.concatenate(tmp, axis=0)
                elif self.output_fmt == 'list':
                    tmp = [vv for v in tmp for vv in v]
            return tmp

    def encode_async(self, batch_generator, max_num_batch=None, delay=0.1, **kwargs):

        def run():
            cnt = 0
            for texts in batch_generator:
                self.encode(texts, blocking=False, **kwargs)
                cnt += 1
                if max_num_batch and cnt == max_num_batch:
                    break

        t = threading.Thread(target=run)
        t.start()
        return self.fetch(delay)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ACManager():
    def __init__(self, available_ac):
        self.available_ac = available_ac
        self.ac = None

    def __enter__(self):
        self.ac = self.available_ac.pop()
        return self.ac

    def __exit__(self, *args):
        self.available_ac.append(self.ac)


class ConcurrentAlpacaClient(AlpacaClient):
    def __init__(self, max_concurrency=10, **kwargs):
        self.available_ac = [AlpacaClient(**kwargs) for _ in range(max_concurrency)]
        self.max_concurrency = max_concurrency

    def close(self):
        for ac in self.available_ac:
            ac.close()

    def _concurrent(func):
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            try:
                with ACManager(self.available_ac) as ac:
                    f = getattr(ac, func.__name__)
                    r = f if isinstance(f, dict) else f(*args, **kwargs)
                return r
            except IndexError:
                raise RuntimeError('Too many concurrent connections!'
                                   'Try to increase the value of "max_concurrency", '
                                   'currently =%d' % self.max_concurrency)

        return arg_wrapper

    @_concurrent
    def encode(self, **kwargs):
        pass

    @property
    @_concurrent
    def server_status(self):
        pass

    @property
    @_concurrent
    def status(self):
        pass

    def fetch(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentBertClient" is not implemented yet')

    def fetch_all(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentBertClient" is not implemented yet')

    def encode_async(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentBertClient" is not implemented yet')
