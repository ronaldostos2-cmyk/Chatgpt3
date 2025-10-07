
import os, time, threading, numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib
from src.model_registry import ModelRegistry
from src.feature_pipeline import compute_features_from_candles
from src import persistence

MIN_BATCH_SIZE = int(os.getenv('AUTO_MIN_BATCH', '128'))
RETRAIN_INTERVAL = int(os.getenv('AUTO_RETRAIN_SECONDS', '300'))

class OnlineBuffer:
    def __init__(self, maxlen=5000):
        self.maxlen = maxlen
        self.data = []
        self.lock = threading.Lock()

    def add(self, features, label):
        with self.lock:
            self.data.append((features, label))
            if len(self.data) > self.maxlen:
                self.data = self.data[-self.maxlen:]

    def sample_batch(self, batch_size):
        with self.lock:
            if len(self.data) < batch_size:
                return None
            idx = np.random.choice(len(self.data), size=batch_size, replace=False)
            X = np.vstack([self.data[i][0] for i in idx])
            y = np.array([self.data[i][1] for i in idx])
            return X, y

    def size(self):
        with self.lock:
            return len(self.data)

class AutoML:
    def __init__(self):
        persistence.init_db()
        self.buffer = OnlineBuffer()
        self.registry = ModelRegistry()
        active = self.registry.get_active()
        if active is None:
            self.model = SGDClassifier(loss='log', max_iter=1000)
            self._initialized = False
        else:
            self.model = active
            self._initialized = True
        self._stop = False
        self._thread = threading.Thread(target=self._retrain_loop, daemon=True)
        self._thread.start()

    def add_observation(self, candles_df: pd.DataFrame, label: int):
        features = compute_features_from_candles(candles_df)
        # persist observation to sqlite
        try:
            persistence.insert_observation(symbol=os.getenv('SYMBOL','BTCUSDT'), features=features.flatten(), label=label)
        except Exception:
            pass
        self.buffer.add(features, label)

    def _retrain_loop(self):
        while not self._stop:
            try:
                if self.buffer.size() >= MIN_BATCH_SIZE:
                    batch = self.buffer.sample_batch(MIN_BATCH_SIZE)
                    if batch is not None:
                        X, y = batch
                        if hasattr(self.model, 'partial_fit'):
                            if not self._initialized:
                                self.model.partial_fit(X, y, classes=np.array([-1,0,1]))
                                self._initialized = True
                            else:
                                self.model.partial_fit(X, y)
                            val = self.buffer.sample_batch(min(256, self.buffer.size()))
                            if val is not None:
                                Xv, yv = val
                                preds = self.model.predict(Xv)
                                acc = accuracy_score(yv, preds)
                                metrics = {'accuracy': float(acc), 'n_samples': int(self.buffer.size())}
                                # persist model metadata and register
                                entry = self.registry.register(self.model, metrics)
                                try:
                                    persistence.insert_model(entry['name'], entry['path'], entry['metrics'])
                                except Exception:
                                    pass
                        else:
                            clf = RandomForestClassifier(n_estimators=100)
                            clf.fit(X, y)
                            val = self.buffer.sample_batch(min(256, self.buffer.size()))
                            if val is not None:
                                Xv, yv = val
                                preds = clf.predict(Xv)
                                acc = accuracy_score(yv, preds)
                                metrics = {'accuracy': float(acc), 'n_samples': int(self.buffer.size())}
                                entry = self.registry.register(clf, metrics)
                                try:
                                    persistence.insert_model(entry['name'], entry['path'], entry['metrics'])
                                except Exception:
                                    pass
                                self.model = clf
                                self._initialized = True
                time.sleep(RETRAIN_INTERVAL)
            except Exception as e:
                print('Erro no loop de retrain', e)
                time.sleep(5)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1)

    def get_model(self):
        return self.registry.get_active()
