def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = " "
    from alpaca_serving import AlpacaServer
    from alpaca_serving.helper import get_run_args
    with AlpacaServer(get_run_args()) as server:
        server.join()