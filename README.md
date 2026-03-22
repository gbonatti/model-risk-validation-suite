# Model Risk Validation Suite

Uma suíte técnica em Python para o desenvolvimento, monitoramento e validação de modelos financeiros (**Independent Model Validation - IMV**). 

Este repositório reúne implementações práticas de metodologias exigidas pelas normas financeiras (**IFRS 9, Basileia III, Bacen**), cobrindo riscos de Crédito, Mercado, Liquidez e Auditoria de Dados.

---

## 🎯 Escopo do Projeto

* **Execução de Validação:** Aplicar testes estatísticos para medir a robustez de modelos preditivos.
* **Monitoramento de Performance:** Acompanhar a estabilidade dos modelos (PSI) e realizar backtesting periódico.
* **Data Quality:** Auditar a integridade das bases de dados utilizadas em modelagem.
* **Explicabilidade:** Utilizar técnicas de SHAP para interpretar as saídas de modelos de Machine Learning.
* **Stress Testing:** Simular impactos de cenários macroeconômicos em indicadores de capital e liquidez.

---

## 📂 Estrutura do Repositório

```text
model-risk-validation-suite/
├── data/                                 # Datasets sintéticos e artefatos de saída
├── src/                                  # Scripts de execução técnica
│   ├── 01_data_generation.py             # Preparação de dados e testes de DQ
│   ├── 02_model_development.py           # Treinamento de modelos (Logística e XGBoost)
│   ├── 03_model_validation.py            # Testes de discriminação (KS, Gini)
│   ├── 04_model_monitoring.py            # Cálculo de estabilidade populacional (PSI)
│   ├── 05_ifrs9_ecl_calculation.py       # Cálculo de Perda Esperada (ECL)
│   ├── 06_irrbb_eve_simulation.py        # Simulação de risco de taxa de juro (EVE)
│   ├── 07_liquidity_risk_lcr.py          # Monitoramento de liquidez (LCR)
│   ├── 08_financial_products_pricing.py  # Auditoria de precificação (Black-Scholes)
│   ├── 09_model_explainability_shap.py   # Interpretação de modelos com SHAP
│   ├── 10_market_risk_var_backtesting.py # Backtesting de VaR (Kupiec)
│   ├── 11_validation_dashboard_plots.py  # Visualização de métricas técnicas
│   ├── 12_model_inventory_manager.py     # Apoio à organização do inventário de modelos
│   ├── 13_backtesting_loss_comparison.py # Comparação entre estimado vs. observado
│   ├── 14_irrbb_nii_sensitivity.py       # Sensibilidade da margem financeira (NII)
│   ├── 15_liquidity_cashflow_stress.py   # Teste de estresse de fluxo de caixa
│   └── 16_fraud_detection_benford.py     # Detecção de anomalias estatísticas (PLD)
├── tests/                                # Auditoria Automatizada
│   └── test_model_assumptions.py         # Testes unitários de premissas (Pytest)
└── README.md                             # Documentação técnica

📝 Resumo das Atividades Técnicas
🛠️ Modelagem & Crédito
01 a 03: Geração de safras, tratamento de nulos, treinamento de algoritmos e validação de poder preditivo (Gini/KS).

05 e 13: Execução do cálculo de ECL (IFRS 9) e realização de backtesting para verificar a aderência das perdas estimadas.

📊 Mercado, Liquidez & Monitoramento
04, 06 e 14: Monitoramento de data drift e análise de sensibilidade do balanço a choques de juros (EVE e NII).

07 e 15: Acompanhamento de indicadores de liquidez e simulação de sobrevivência de caixa em estresse.

10: Validação estatística de modelos de VaR para risco de mercado.

🔍 Auditoria e Interpretação
09: Tradução das variáveis do XGBoost para garantir que o modelo segue premissas econômicas.

16: Aplicação da Lei de Benford para identificar desvios em bases transacionais.

tests/: Rotinas de teste automatizado para garantir que os modelos respeitam limites normativos.

⚙️ Instalação e Execução
Instale as dependências:

pip install pandas numpy scikit-learn xgboost scipy joblib shap matplotlib seaborn pytest
Execute os testes de integridade:

python -m pytest tests/ -v
👨‍💻 Autor
Gilberto Ricardo Bonatti - Analista de Validação de Modelos e Dados