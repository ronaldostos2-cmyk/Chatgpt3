
# script de treino simples para gerar modelo baseline
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

N = 2000
np.random.seed(42)
X = np.random.randn(N, 4)
y = np.random.choice([-1, 0, 1], size=N, p=[0.3, 0.4, 0.3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
joblib.dump(clf, models_dir / 'baseline_rf.joblib')
print('Modelo salvo em models/baseline_rf.joblib')
