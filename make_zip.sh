
#!/usr/bin/env bash
set -e
ZIPNAME=bot-binance-testnet.zip
rm -f $ZIPNAME
zip -r $ZIPNAME . -x ".git/*" "venv/*" "__pycache__/*" "$ZIPNAME"
echo "Gerado $ZIPNAME"
