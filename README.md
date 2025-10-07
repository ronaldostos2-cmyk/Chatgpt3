
# Bot Binance Futures Testnet (Scaffold + AutoML)

Scaffold inicial com módulos básicos, adapter para Binance Testnet Futures, Risk Manager,
AI baseline e Auto-ML (online buffer + retrain).

**Como usar**:
1. Copie `.env.example` para `.env` e preencha as chaves Testnet.
2. Crie virtualenv e instale dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Treine modelo baseline (opcional):
   ```bash
   python src/train.py
   ```
4. Rode o bot em Testnet:
   ```bash
   python run.py
   ```
