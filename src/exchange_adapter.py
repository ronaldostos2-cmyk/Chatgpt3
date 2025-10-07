
import os, time, math, logging
from threading import Thread
from queue import Queue, Empty
from binance.client import Client
from binance import ThreadedWebsocketManager

logger = logging.getLogger('exchange_adapter')

class ExchangeAdapter:
    def __init__(self):
        api_key = os.getenv('BINANCE_TESTNET_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_SECRET')
        if not api_key or not api_secret:
            logger.error('API Key/Secret não configuradas. Verifique .env')
            raise RuntimeError('Chaves da Testnet não configuradas')
        self.client = Client(api_key, api_secret, testnet=True)
        self.order_queue = Queue()
        self._worker_thread = Thread(target=self._order_worker, daemon=True)
        self._worker_thread.start()
        self._symbol_info = {}
        try:
            self._load_exchange_info()
        except Exception as e:
            logger.warning('Falha ao carregar exchange info: %s', e)
        self.twm = None

    def _load_exchange_info(self):
        info = self.client.futures_exchange_info()
        for s in info.get('symbols', []):
            self._symbol_info[s['symbol']] = s
        logger.info('exchange_info carregada com %d symbols', len(self._symbol_info))

    def _get_symbol_filters(self, symbol: str):
        s = self._symbol_info.get(symbol)
        if not s:
            self._load_exchange_info()
            s = self._symbol_info.get(symbol)
            if not s:
                raise ValueError(f'Symbol info não encontrada: {symbol}')
        return s.get('filters', [])

    def get_price(self, symbol: str) -> float:
        tick = self.client.futures_symbol_ticker(symbol=symbol)
        return float(tick['price'])

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        filters = self._get_symbol_filters(symbol)
        step = None
        for f in filters:
            if f['filterType'] == 'LOT_SIZE':
                step = float(f['stepSize'])
                break
        if step is None:
            return quantity
        precision = int(round(-math.log(step, 10))) if step < 1 else 0
        rounded = math.floor(quantity / step) * step
        return float(round(rounded, precision))

    def place_market_order(self, symbol: str, side: str, quantity: float, max_retries: int = 3):
        qty = float(quantity)
        order = {'symbol': symbol, 'side': side, 'quantity': qty, 'retries': 0, 'max_retries': max_retries}
        self.order_queue.put(order)
        return {'status': 'queued', 'symbol': symbol, 'side': side, 'quantity': qty}

    def _order_worker(self):
        while True:
            try:
                order = self.order_queue.get(timeout=1)
            except Empty:
                time.sleep(0.1)
                continue
            symbol = order['symbol']
            side = order['side']
            qty = order['quantity']
            try:
                qty_norm = self._round_quantity(symbol, qty)
                if qty_norm <= 0:
                    raise ValueError('Quantidade normalizada ficou <= 0')
                res = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=qty_norm
                )
                logger.info('Ordem executada: %s %s %s', symbol, side, qty_norm)
            except Exception as e:
                order['retries'] += 1
                logger.exception('Erro ao enviar ordem (tentativa %d): %s', order['retries'], e)
                if order['retries'] <= order['max_retries']:
                    time.sleep(1 + order['retries'] * 1.5)
                    self.order_queue.put(order)
                else:
                    logger.error('Ordem falhou após %d tentativas: %s', order['retries'], e)

    def get_position(self, symbol: str):
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            return pos
        except Exception as e:
            logger.exception('Erro ao obter posição')
            return None

    def start_candles_ws(self, symbol: str, interval: str, callback):
        if self.twm is None:
            self.twm = ThreadedWebsocketManager(api_key=self.client.API_KEY, api_secret=self.client.API_SECRET, testnet=True)
            self.twm.start()
        def _cb(msg):
            try:
                callback(msg)
            except Exception as e:
                logger.exception('Erro no callback do websocket: %s', e)
        self.twm.start_kline_socket(callback=_cb, symbol=symbol, interval=interval)

    def stop(self):
        if self.twm:
            try:
                self.twm.stop()
            except Exception:
                pass
