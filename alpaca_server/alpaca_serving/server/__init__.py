def main():
    from alpaca_server import AlpacaServer
    from alpaca_server.helper import get_run_args
    with AlpacaServer(get_run_args()) as server:
        server.join()


# def benchmark():
#     from bert_serving.server.benchmark import run_benchmark
#     from bert_serving.server.helper import get_run_args, get_benchmark_parser
#     args = get_run_args(get_benchmark_parser)
#     run_benchmark(args)


def terminate():
    from alpaca_server import AlpacaServer
    from alpaca_server.helper import get_run_args, get_shutdown_parser
    args = get_run_args(get_shutdown_parser)
    AlpacaServer.shutdown(args)