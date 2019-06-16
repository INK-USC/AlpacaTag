import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from alpaca_server.alpaca_serving import AlpacaServer
from alpaca_server.alpaca_serving.helper import get_run_args
with AlpacaServer(get_run_args()) as server:
    server.join()