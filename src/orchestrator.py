
import os, logging, time
from dotenv import load_dotenv
from src.logging_config import setup_logging
from src.metrics_server import start_metrics
from src.exchange_adapter import ExchangeAdapter
from src.risk_manager import RiskManager
from src.ai_signal import AISignal
from src.auto_ml import AutoML

load_dotenv()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
setup_logging(LOG_LEVEL)
import logging
logger = logging.getLogger('orchestrator')
start_metrics(int(os.getenv('METRICS_PORT', '8000')))


class Orchestrator:
    def __init__(self):
        self.exchange = ExchangeAdapter()
        self.risk = RiskManager()
        self.signal = AISignal()
        self.auto_ml = AutoML()

    def start(self):
        logger.info('Orchestrator iniciado (modo Testnet)')
        sig = self.signal.get_signal()
        logger.info(f'Sinal recebido: {sig}')
        if sig['action'] in ('BUY', 'SELL'):
            qty = sig['quantity']
            qty_ok = self.risk.normalize_quantity(sig['symbol'], qty)
            if not self.risk.check_exposure(sig['symbol'], qty_ok):
                logger.warning('Ordem bloqueada pelo Risk Manager')
                return
            res = self.exchange.place_market_order(sig['symbol'], sig['action'], qty_ok)
            logger.info(f'Order response: {res}')
            # exemplo simples de rotulagem: espera 30s e pega preço para label
            try:
                exec_price = None
                time.sleep(5)  # aguarda breve
                price_now = self.exchange.get_price(sig['symbol'])
                # label: lucro se price moved favorably by threshold (placeholder)
                # para demo, label 0 (neutral)
                label = 0
                # collecte candles e envie para auto_ml buffer (se disponível)
                candles = None
                try:
                    klines = self.exchange.client.futures_klines(symbol=sig['symbol'], interval='1m', limit=60)
                    import pandas as pd
                    df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
                    df = df[['open','high','low','close','volume']].astype(float)
                    self.auto_ml.add_observation(df, label)
                except Exception:
                    pass
        else:
            logger.info('Nenhuma ação tomada (signal neutro)')
