import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

df = pd.read_parquet('../data/credit_portfolio.parquet')
df['renda_mensal'] = df['renda_mensal'].fillna(df['renda_mensal'].median())
df['score_bureau'] = df['score_bureau'].replace(-999, df['score_bureau'].median())

features = ['idade', 'renda_mensal', 'score_bureau', 'valor_emprestimo', 'taxa_juros', 'LTV']
X = df[features]
y = df['default_12m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, '../data/xgb_pd_model.pkl')
joblib.dump((X_test, y_test), '../data/test_data.pkl')
print("Modelo desenvolvido e artefatos salvos.")
