# Model Risk Management & Validation Suite

Uma suite abrangente e automatizada em Python para o desenvolvimento, monitorização e validação independente de modelos financeiros (Independent Model Validation - IMV). 

Este repositório serve como um portfólio prático das metodologias exigidas pelas principais normas financeiras e regulatórias (IFRS 9, Basileia III, Bacen), cobrindo múltiplos domínios de risco: Crédito, Mercado, Liquidez e Precificação de Produtos Financeiros.

## 🎯 Objetivos do Projeto

- **Validação de Modelos:** Avaliar a robustez matemática e estatística de modelos preditivos e estruturais, garantindo a ausência de enviesamentos e a estabilidade ao longo do tempo.
- **Conformidade Regulatória (Compliance):** Assegurar que os modelos operam dentro dos limites estabelecidos por normas internas e externas.
- **Qualidade de Dados (Data Quality):** Auditar a integridade, fiabilidade e disponibilidade dos dados utilizados nas fases de desenvolvimento e operação.
- **Testes de Stress & Limites:** Aplicar choques macroeconómicos e testar o comportamento assintótico das equações de precificação.
- **Explicabilidade (XAI):** Garantir que modelos complexos (como XGBoost) respeitam a lógica económica e regulatória através de técnicas como SHAP.

## 📂 Estrutura do Repositório

```text
model-risk-validation-suite/
│
├── data/                                 # Datasets sintéticos e artefactos (modelos .pkl, gráficos .png)
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
│   └── 11_validation_dashboard_plots.py  # Geração do Dashboard de Validação Visual
│
├── tests/                                # Testes Unitários e Validação de Premissas
│   └── test_model_assumptions.py         # Testes rigorosos em Pytest para auditoria de limites
│
└── README.md                             # Documentação do projeto


📝 Resumo dos Scripts
Risco de Crédito & IFRS 9
01_data_generation.py: Gera um dataset sintético focado em crédito. Calcula a probabilidade de default real (Target) e introduz ruídos/nulos para simular problemas de Data Quality e testar a resiliência do pipeline.

02_model_development.py: Treina os modelos de Probabilidade de Default (PD) usando Regressão Logística (benchmark regulatório) e XGBoost.

03_model_validation.py: Validação de performance focada em discriminação, calculando Gini, Curva ROC e KS.

04_model_monitoring.py: Monitorização contínua usando o Population Stability Index (PSI) para capturar a degradação e instabilidade do modelo em novas safras.

05_ifrs9_ecl_calculation.py: Calcula a Perda de Crédito Esperada (ECL) integrando as premissas de PD, Loss Given Default (LGD) e Exposure at Default (EAD).

Risco de Mercado, Liquidez e Precificação
06_irrbb_eve_simulation.py: Simula choques paralelos na curva de taxas de juros e audita o impacto no Valor Económico do Património (EVE).

07_liquidity_risk_lcr.py: Valida as premissas de escoamento de depósitos (run-off rates) sob cenários de stress severo para auditar o indicador LCR.

08_financial_products_pricing.py: Audita limites matemáticos e sensibilidades ("Gregas") de modelos analíticos de derivativos como o Black-Scholes.

10_market_risk_var_backtesting.py: Aplica o Teste de Kupiec para garantir que as exceções históricas do modelo de Value at Risk (VaR) estão contidas dentro do limite estatístico aceitável.

Explicabilidade, Auditoria e Apresentação
09_model_explainability_shap.py: Desconstrói o modelo XGBoost usando valores SHAP para validar a monotonicidade das variáveis, garantindo que relações como "juros altos = risco alto" são aprendidas corretamente.

11_validation_dashboard_plots.py: Consolida as métricas críticas num Dashboard Visual gerado automaticamente, cobrindo a ROC, Calibração, Estabilidade Temporal (PSI) e impactos no IRRBB.

test_model_assumptions.py (Pasta tests/): Suíte de auditoria automatizada em pytest que funciona como um validador algorítmico, estressando os axiomas e as condições de limite normativo de todos os modelos acima.

⚙️ Instalação e Execução
Para rodar todo o portfólio localmente, é recomendada a utilização de um ambiente virtual Python (versão 3.8+).

Instale as bibliotecas e dependências:

Bash
pip install pandas numpy scikit-learn xgboost scipy joblib shap matplotlib seaborn pytest
Navegue até a pasta de código-fonte e execute a esteira sequencialmente:

Bash
cd src
python 01_data_generation.py
python 02_model_development.py
# ... e assim sucessivamente para os outros scripts
Para executar a Suíte de Auditoria Independente:
A partir da raiz do projeto, corra:

Bash
pytest tests/ -v
👨‍💻 Autor
Gilberto Ricardo Bonatti Analista Sénior de Validação de Modelos e Risco
