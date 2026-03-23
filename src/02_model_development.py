import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

df = pd.read_parquet('../data/credit_portfolio.parquet')
df['renda_mensal'] = df['renda_mensal'].fillna(df['renda_mensal'].median())
df['score_bureau'] = df['score_bureau'].replace(-999, df['score_bureau'].median())

features = ['idade', 'renda_mensal', 'score_bureau', 'valor_emprestimo', 'taxa_juros', 'LTV']
X = df[features]
y = df['default_12m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── Modelo 1: Regressão Logística (benchmark regulatório) ──────────────────
# Escalonamento necessário para convergência e coeficientes interpretáveis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_model.fit(X_train_scaled, y_train)

joblib.dump((lr_model, scaler), '../data/lr_pd_model.pkl')
print("Regressão Logística treinada e salva em '../data/lr_pd_model.pkl'")

# ── Modelo 2: XGBoost (modelo candidato) ──────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, '../data/xgb_pd_model.pkl')
joblib.dump((X_test, y_test), '../data/test_data.pkl')
joblib.dump((X_test_scaled, y_test), '../data/test_data_scaled.pkl')

print("XGBoost treinado e salvo em '../data/xgb_pd_model.pkl'")
print("Artefatos de teste salvos com sucesso.")
