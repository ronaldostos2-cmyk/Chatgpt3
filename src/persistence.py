
import sqlite3
from pathlib import Path
from typing import Optional, Any, List, Tuple
import threading
import json

DB_PATH = Path('data/bot_data.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()

def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        # observations: id, timestamp, symbol, features (json), label
        cur.execute(
"""
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT,
    features_json TEXT,
    label INTEGER
);
"""
        )
        # models: id, timestamp, name, path, metrics_json
        cur.execute(
"""
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    name TEXT,
    path TEXT,
    metrics_json TEXT
);
"""
        )
        conn.commit()
        conn.close()

def insert_observation(symbol: str, features: Any, label: int):
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO observations (symbol, features_json, label) VALUES (?, ?, ?)",
                    (symbol, json.dumps(features.tolist() if hasattr(features, 'tolist') else features), label))
        conn.commit()
        conn.close()

def fetch_observations(limit: int = 1000) -> List[sqlite3.Row]:
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM observations ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows

def insert_model(name: str, path: str, metrics: dict):
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO models (name, path, metrics_json) VALUES (?, ?, ?)",
                    (name, path, json.dumps(metrics)))
        conn.commit()
        conn.close()

def list_models(limit: int = 100) -> List[sqlite3.Row]:
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM models ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
