model-risk-validation-suite/
│
├── data/                                 # Datasets sintéticos, Inventário (.csv) e gráficos (.png)
├── src/                                  # Código-fonte principal
│   ├── 01_data_generation.py             # Geração de dados de crédito e testes de Data Quality
│   ├── 02_model_development.py           # Treino de modelos (Regressão Logística e XGBoost)
│   ├── 03_model_validation.py            # Validação estatística (KS, Gini, Calibração)
│   ├── 04_model_monitoring.py            # Monitorização de degradação via PSI (Data Drift)
│   ├── 05_ifrs9_ecl_calculation.py       # Cálculo da Perda Esperada de Crédito (PD, LGD, EAD)
│   ├── 06_irrbb_eve_simulation.py        # Risco de Taxa de Juro e simulação do EVE
│   ├── 07_liquidity_risk_lcr.py          # Validação de Risco de Liquidez (LCR sob stress)
│   ├── 08_financial_products_pricing.py  # Precificação de Derivativos (Black-Scholes e Gregas)
│   ├── 09_model_explainability_shap.py   # Validação de Explicabilidade com Valores SHAP
│   ├── 10_market_risk_var_backtesting.py # Backtesting de VaR (Teste de Kupiec)
│   ├── 11_validation_dashboard_plots.py  # Geração do Dashboard de Validação Visual
│   ├── 12_model_inventory_manager.py     # Gestão de Inventário Centralizado e Materialidade
│   └── 13_backtesting_loss_comparison.py # Comparação Perdas Estimadas vs. Realizadas
│
├── tests/                                # Testes Unitários e Validação de Premissas
│   └── test_model_assumptions.py         # Testes rigorosos em Pytest para auditoria de limites
│
└── README.md                             # Documentação do projeto


📝 Resumo dos Scripts
Governança e Inventário
12_model_inventory_manager.py: Implementa a manutenção de um inventário centralizado de modelos, classificando-os por materialidade, risco e status de validação, conforme exigido pelas boas práticas de Governança de Risco de Modelo (MRM).

Risco de Crédito & IFRS 9
01_data_generation.py: Gera dataset sintético de crédito. Calcula a probabilidade de default real e introduz ruídos/nulos para testar a resiliência do pipeline a falhas de dados.

02_model_development.py: Treina modelos de PD (Regressão Logística e XGBoost).

03_model_validation.py: Validação de performance (Gini, Curva ROC e KS).

05_ifrs9_ecl_calculation.py: Cálculo da Perda de Crédito Esperada (ECL) integrando PD, LGD e EAD.

13_backtesting_loss_comparison.py: Realiza a avaliação periódica de desempenho comparando as perdas estimadas pelo modelo IFRS 9 contra as perdas efetivamente observadas na operação.

Risco de Mercado, Liquidez e Monitoramento
04_model_monitoring.py: Monitorização contínua via PSI (Population Stability Index) para identificar desvios populacionais.

06_irrbb_eve_simulation.py: Simula choques na curva de juros para auditar o impacto no EVE.

07_liquidity_risk_lcr.py: Valida o indicador LCR sob cenários de stress de liquidez.

10_market_risk_var_backtesting.py: Aplica o Teste de Kupiec para validar exceções de modelos de VaR.

Explicabilidade, Auditoria e Apresentação
08_financial_products_pricing.py: Audita limites matemáticos e "Gregas" em precificação de derivativos.

09_model_explainability_shap.py: Utiliza SHAP para garantir que o modelo de ML respeita a lógica económica e não apresenta comportamentos de "caixa preta".

11_validation_dashboard_plots.py: Consolida métricas em um Dashboard visual para reporte em Comitês de Risco.

tests/test_model_assumptions.py: Suíte de auditoria automatizada em pytest que estressa axiomas e condições de limite normativo.

⚙️ Instalação e Execução
Para rodar todo o portfólio localmente, é recomendada a utilização de um ambiente virtual Python (versão 3.8+).

Instale as bibliotecas e dependências:

Bash
pip install pandas numpy scikit-learn xgboost scipy joblib shap matplotlib seaborn pytest
Execute a esteira sequencialmente (dentro da pasta src):

Bash
cd src
python 01_data_generation.py
python 02_model_development.py
# ... execute os demais conforme necessário
Para executar a Suíte de Auditoria Independente:
A partir da raiz do projeto, corra:

Bash
python -m pytest tests/ -v
👨‍💻 Autor
Gilberto Ricardo Bonatti
