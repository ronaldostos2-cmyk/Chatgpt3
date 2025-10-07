
import os, logging, joblib
from pathlib import Path

logger = logging.getLogger('ai_signal')
MODEL_PATH = Path(os.getenv('MODEL_PATH', 'models/baseline_rf.joblib'))

class AISignal:
    def __init__(self):
        self.model = None
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info('Modelo carregado: %s', MODEL_PATH)
            except Exception as e:
                logger.exception('Falha ao carregar modelo: %s', e)

    def get_signal(self):
        symbol = os.getenv('SYMBOL', 'BTCUSDT')
        quantity = float(os.getenv('QUANTITY', '0.001'))
        if self.model is None:
            return {'symbol': symbol, 'action': 'HOLD', 'quantity': quantity}
        X = [[0.0, 0.0, 0.0, 0.0]]
        pred = self.model.predict(X)[0]
        action = 'HOLD'
        if pred == 1:
            action = 'BUY'
        elif pred == -1:
            action = 'SELL'
        return {'symbol': symbol, 'action': action, 'quantity': quantity}
