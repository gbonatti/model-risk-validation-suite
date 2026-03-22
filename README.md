model-risk-validation-suite/
├── data/                                 # Datasets, Inventário (.csv) e plots (.png)
├── src/                                  # Código-fonte principal
│   ├── 01_data_generation.py             # Geração de dados e Data Quality
│   ├── 02_model_development.py           # Treino de modelos (Reg. Logística e XGBoost)
│   ├── 03_model_validation.py            # Validação estatística (KS, Gini, Calibração)
│   ├── 04_model_monitoring.py            # Monitorização de degradação (PSI)
│   ├── 05_ifrs9_ecl_calculation.py       # Cálculo de Perda Esperada (ECL)
│   ├── 06_irrbb_eve_simulation.py        # Risco de Taxa de Juro (EVE)
│   ├── 07_liquidity_risk_lcr.py          # Risco de Liquidez (LCR)
│   ├── 08_financial_products_pricing.py  # Precificação e "Gregas" (Black-Scholes)
│   ├── 09_model_explainability_shap.py   # Explicabilidade com Valores SHAP
│   ├── 10_market_risk_var_backtesting.py # Backtesting de VaR (Teste de Kupiec)
│   ├── 11_validation_dashboard_plots.py  # Dashboard Visual de Validação
│   ├── 12_model_inventory_manager.py     # Gestão de Inventário e Materialidade
│   └── 13_backtesting_loss_comparison.py # Comparação Estimado vs. Realizado
├── tests/                                # Auditoria Automatizada
│   └── test_model_assumptions.py         # Testes rigorosos em Pytest
└── README.md                             # Documentação do projeto

📝 Resumo dos Scripts
🏛️ Governança e Inventário
12_model_inventory_manager.py: Mantém o inventário centralizado, classificando modelos por materialidade e status de validação (requisito de Governança de Risco de Modelo - MRM).

💳 Risco de Crédito & IFRS 9
01_data_generation.py: Gera dados sintéticos e injeta falhas propositais para testar a resiliência do modelo.

02_model_development.py: Compara modelos tradicionais e de ML para Probabilidade de Default.

03_model_validation.py: Valida o poder discriminatório (Gini e KS).

05_ifrs9_ecl_calculation.py: Calcula a provisão de perda esperada (ECL).

13_backtesting_loss_comparison.py: Avalia o desempenho real comparando a perda estimada vs. a inadimplência observada.

📈 Risco de Mercado, Liquidez e Monitoramento
04_model_monitoring.py: Detecta desvios populacionais (PSI).

06_irrbb_eve_simulation.py: Mede a sensibilidade do património a choques na curva de juros.

07_liquidity_risk_lcr.py: Audita a solvência em cenários de stress de 30 dias.

10_market_risk_var_backtesting.py: Valida estatisticamente o modelo de VaR.

🔍 Explicabilidade e Apresentação
09_model_explainability_shap.py: Garante que o modelo de ML não é uma "caixa preta", validando a lógica económica das variáveis.

11_validation_dashboard_plots.py: Gera o Dashboard visual para reporte executivo.

tests/test_model_assumptions.py: Auditoria "robótica" que estressa os axiomas matemáticos dos modelos.

⚙️ Instalação e Execução
Instale as dependências:

Bash
pip install pandas numpy scikit-learn xgboost scipy joblib shap matplotlib seaborn pytest
Execute a esteira (dentro da pasta src):

Bash
cd src
python 01_data_generation.py
python 02_model_development.py
# Execute os demais conforme necessário
Para executar a Auditoria Independente:
A partir da raiz do projeto, corra:

Bash
python -m pytest tests/ -v
👨‍💻 Autor
Gilberto Ricardo Bonatti