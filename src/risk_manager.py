
import os, logging
from src.exchange_adapter import ExchangeAdapter

logger = logging.getLogger('risk_manager')

class RiskManager:
    def __init__(self):
        self.max_notional = float(os.getenv('MAX_NOTIONAL', '500'))
        self.max_leverage = float(os.getenv('MAX_LEVERAGE', '10'))
        self.exchange = ExchangeAdapter()

    def normalize_quantity(self, symbol: str, quantity: float) -> float:
        try:
            qty = float(quantity)
            qty_norm = self.exchange._round_quantity(symbol, qty)
            return qty_norm
        except Exception as e:
            logger.exception('Erro ao normalizar quantity: %s', e)
            return 0.0

    def check_exposure(self, symbol: str, quantity: float) -> bool:
        try:
            price = self.exchange.get_price(symbol)
            notional = price * quantity
            logger.info('Notional estimado: %.2f USDT (limite %.2f)', notional, self.max_notional)
            if notional > self.max_notional:
                logger.warning('Bloqueado: notional acima do limite')
                return False
            account = self.exchange.client.futures_account_balance()
            usdt_balance = 0.0
            for a in account:
                if a['asset'] == 'USDT':
                    usdt_balance = float(a['balance'])
                    break
            estimated_exposure = notional / max(1.0, self.max_leverage)
            if estimated_exposure > usdt_balance:
                logger.warning('Bloqueado: exposição estimada excede saldo disponível')
                return False
            return True
        except Exception as e:
            logger.exception('Erro ao checar exposure: %s', e)
            return False
