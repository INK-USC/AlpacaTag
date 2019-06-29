import multiprocessing
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
import os.path
import json

import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from alpaca_server.alpaca_serving.helper import *
from alpaca_server.alpaca_serving.httpproxy import HTTPProxy
from alpaca_server.alpaca_serving.zmq_decor import multi_socket
from alpaca_server.alpaca_model.pytorchAPI import SequenceTaggingModel

__all__ = ['__version__']
__version__ = '1.0.1'


class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'

    initiate = b'INITIATE'
    online_initiate = b'ONLINE_INITIATE'
    online_learning = b'ONLINE_LEARNING'
    active_learning = b'ACTIVE_LEARNING'
    predict = b'PREDICT'
    load = b'LOAD'
    error = b'ERROR'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


# Ventilator
# Ventilator pushes data to workers with PUSH pattern.
class AlpacaServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'))
        self.args = args

        # ZeroMQ server configuration
        self.num_worker = args.num_worker  # number of Workers

        # restrict number of workers for temporaly
        self.num_concurrent_socket = max(16, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port

        # project configuration
        self.model_dir = args.model_dir  # alpaca_model per project
        self.model = None # pass this model to every sink and worker!!!!
        # learning initial configuration
        self.batch_size = args.batch_size
        self.epoch = args.epoch

        self.status_args = {k: v for k, v in sorted(vars(args).items())}
        self.status_static = {
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }

        self.processes = []
        self.logger.info('Initialize the alpaca_model... could take a while...')
        self.is_ready = threading.Event()

    def __enter__(self):
        self.start()
        self.is_ready.wait()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        self.is_ready.clear()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'0', ServerCmd.terminate, b'0', b'0'])

    @staticmethod
    def shutdown(args):
        with zmq.Context() as ctx:
            ctx.setsockopt(zmq.LINGER, args.timeout)
            with ctx.socket(zmq.PUSH) as frontend:
                try:
                    frontend.connect('tcp://%s:%d' % (args.ip, args.port))
                    frontend.send_multipart([b'0', ServerCmd.terminate, b'0', b'0'])
                    print('shutdown signal sent to %d' % args.port)
                except zmq.error.Again:
                    raise TimeoutError(
                        'no response from the server (with "timeout"=%d ms), please check the following:'
                        'is the server still online? is the network broken? are "port" correct? ' % args.timeout)

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):
        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets' % len(addr_backend_list))

        # start the sink process
        self.logger.info('start the sink')
        proc_sink = AlpacaSink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = AlpacaWorker(idx, self.args, self.model, addr_backend_list, addr_sink, device_id)
            self.processes.append(process)
            process.start()

        # start the http-service process
        if self.args.http_port:
            self.logger.info('start http proxy')
            proc_proxy = HTTPProxy(self.args)
            self.processes.append(proc_proxy)
            proc_proxy.start()

        rand_backend_socket = None
        server_status = ServerStatistic()

        for p in self.processes:
            p.is_ready.wait()

        self.is_ready.set()
        self.logger.info('all set, ready to serve request!')

        # receive message from client
        # make commands (1.recommend, 2.online learning(training) 3.kerasAPI learning ...)
        # project based file management
        while True:
            try:
                request = frontend.recv_multipart()
                client, msg_type, msg, req_id, msg_len = request
                assert req_id.isdigit()
                assert msg_len.isdigit()
            except (ValueError, AssertionError):
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg_type == ServerCmd.terminate:
                    break
                elif msg_type == ServerCmd.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'device_map': device_map,
                                      'num_concurrent_socket': self.num_concurrent_socket}

                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_args,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' % (int(req_id), int(msg_len), client))

                    # register a new job at sink
                    sink.send_multipart([client, ServerCmd.new_job, msg_len, req_id])
                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backend shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job
                    job_id = client + b'#' + req_id

                    try:
                        rand_backend_socket.send_multipart([job_id, msg_type, msg],zmq.NOBLOCK)  # fixed!
                    except zmq.error.Again:
                        self.logger.info('zmq.error.Again: resource not available temporally, please send again!')
                        sink.send_multipart([client, ServerCmd.error, jsonapi.dumps('zmq.error.Again: resource not available temporally, please send again!'), req_id])

        for p in self.processes:
            p.close()
        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker),
                                                maxMemory=0.9, maxLoad=0.9)
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


class AlpacaSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'))
        self.front_sink_addr = front_sink_addr
        self.is_ready = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.is_ready.clear()
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        # have to make jobs.
        # type: Dict[str, SinkJob]
        pending_jobs = defaultdict(lambda: SinkJob(0))

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger = set_logger(colored('SINK', 'green'))
        logger.info('ready')
        self.is_ready.set()

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                # job_id = client + b'#' + req_id
                job_id = msg[0]
                x = msg[1]
                job_type = msg[2]
                # parsing job_id and partial_id
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = int(job_info[1]) if len(job_info) == 2 else 0

                pending_jobs[job_id].add_job(x, job_type, partial_id)
                logger.info('collect %s %s' % (x, job_id))

                # check if there are finished jobs, then send it back to workers
                finished = [(k, v) for k, v in pending_jobs.items() if v.is_done]
                for job_info, tmp in finished:
                    client_addr, req_id = job_info.split(b'#')
                    x = tmp.result
                    sender.send_multipart([client_addr, x, req_id])
                    logger.info('send back\tjob id: %s' % (job_info))
                    # release the job
                    tmp.clear()
                    pending_jobs.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCmd.new_job:
                    job_info = client_addr + b'#' + req_id
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                if msg_type == ServerCmd.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])
                if msg_type == ServerCmd.error:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send error\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, msg_type, req_id])

class SinkJob:
    def __init__(self, req_id):
        self.req_id = req_id
        self.result_msg = None

    def clear(self):
        self.req_id = 0

    def add_job(self, data, job_type, pid):
        if job_type == ServerCmd.initiate:
            self.result_msg = 'Model initiated'
        elif job_type == ServerCmd.online_initiate:
            self.result_msg = 'Online word build completed'
        elif job_type == ServerCmd.online_learning:
            self.result_msg = 'Online learning completed'
        elif job_type == ServerCmd.predict:
            self.result_msg = data
        elif job_type == ServerCmd.load:
            self.result_msg = 'Model Loaded'
        elif job_type == ServerCmd.active_learning:
            self.result_msg = data
    @property
    def is_done(self):
        return True

    @property
    def result(self):
        if self.result_msg is not None:
            if type(self.result_msg) == bytes:
                x_info = self.result_msg
            else:
                x_info = jsonapi.dumps(self.result_msg)
        self.result_msg = None
        return x_info


class AlpacaWorker(Process):
    def __init__(self, id, args, model, worker_address_list, sink_address, device_id):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'))
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.use_fp16 = args.fp16
        self.show_tokens_to_client = args.show_tokens_to_client
        self.is_ready = multiprocessing.Event()

        self.model = model
        self.modelid = 0 #project_id

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.is_ready.clear()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, outputs, inputs, *receivers):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'))

        logger.info('use device %s' %
                    ('cpu' if self.device_id < 0 else 'gpu: %d' % self.device_id))

        poller = zmq.Poller()
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)
            poller.register(sock, zmq.POLLIN)

        outputs.connect(self.sink_address)
        inputs.connect(self.sink_address)

        logger.info('ready and listening!')
        self.is_ready.set()

        while not self.exit_flag.is_set():
            events = dict(poller.poll())
            for sock_idx, sock in enumerate(receivers):
                if sock in events:
                    client_id, msg_type, raw_msg = sock.recv_multipart()
                    msg = jsonapi.loads(raw_msg)

                    if msg_type == ServerCmd.initiate:
                        self.model = SequenceTaggingModel()
                        self.modelid = str(msg)
                        if os.path.isfile(os.path.join('.','model'+self.modelid+'.pre')) and os.path.isfile(os.path.join('.','model'+self.modelid+'.pt')):
                            self.model.load('model'+self.modelid)
                            logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, 1, client_id))
                            helper.send_test(outputs, client_id, b'Model Loaded', ServerCmd.load)
                            logger.info('job done\tsize: %s\tclient: %s' % (1, client_id))
                        else:
                            logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, 1, client_id))
                            helper.send_test(outputs, client_id, b'Model Initiated', msg_type)
                            logger.info('job done\tsize: %s\tclient: %s' % (1, client_id))

                    elif msg_type == ServerCmd.online_initiate:
                        #django side -> directly read from database as the msg[0[ / msg[1] could be so huge.
                        #since we already know the project id.
                        self.model.online_word_build(msg[0],msg[1]) # whole unlabeled training sentences / predefined_labels
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg[0]), client_id))
                        helper.send_test(outputs, client_id, b'Online word build completed', msg_type)
                        logger.info('job done\tsize: %s\tclient: %s' % (len(msg[0]), client_id))

                    elif msg_type == ServerCmd.online_learning:
                        self.model.online_learning(msg[0], msg[1])
                        self.model.save('model'+self.modelid)
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg[0]), client_id))
                        helper.send_test(outputs, client_id, b'Online learning completed', msg_type)
                        logger.info('job done\tsize: %s\tclient: %s' % (len(msg[0]), client_id))

                    elif msg_type == ServerCmd.predict:
                        analyzed_result = self.model.analyze(msg)
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, 1, client_id))
                        helper.send_test(outputs, client_id, jsonapi.dumps(analyzed_result), msg_type)
                        logger.info('job done\tsize: %s\tclient: %s' % (1, client_id))

                    elif msg_type == ServerCmd.active_learning:
                        indices = self.model.active_learning(msg[0], msg[1])
                        json_indices = list(map(int, indices))
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg[0]), client_id))
                        helper.send_test(outputs, client_id, jsonapi.dumps(json_indices), msg_type)
                        logger.info('job done\tsize: %s\tclient: %s' % (len(msg[0]), client_id))


class ServerStatistic:
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client, msg_type, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCmd.is_valid(msg_type):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat):
            if len(stat) > 0:
                return {
                    'avg_%s' % name: sum(stat) / len(stat),
                    'min_%s' % name: min(stat),
                    'max_%s' % name: max(stat),
                    'num_min_%s' % name: sum(v == min(stat) for v in stat),
                    'num_max_%s' % name: sum(v == max(stat) for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client kerasAPI when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': len(self._hist_client),
            'num_active_client': get_num_active_client()},
            get_min_max_avg('request_per_client', self._hist_client.values()),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys()),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}