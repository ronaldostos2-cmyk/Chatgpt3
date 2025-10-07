
from prometheus_client import start_http_server, Counter, Gauge
import threading, time, os

ORDER_COUNTER = Counter('bot_orders_total', 'Number of orders executed', ['symbol','side'])
MODEL_ACCURACY = Gauge('bot_model_accuracy', 'Current model accuracy')

def start_metrics(port:int=8000):
    def _run():
        start_http_server(port)
        while True:
            time.sleep(1)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
