# Model Risk Management & Validation Suite

Uma suíte abrangente e automatizada em Python para o desenvolvimento, monitorização e validação independente de modelos financeiros (**Independent Model Validation - IMV**). 

Este repositório serve como um portfólio prático das metodologias exigidas pelas principais normas financeiras e regulatórias (**IFRS 9, Basileia III, Bacen**), cobrindo múltiplos domínios de risco: Crédito, Mercado, Liquidez e Precificação de Produtos Financeiros.

---

## 🎯 Objetivos do Projeto

* **Governança de Modelos:** Manutenção de inventário centralizado, controle de versões e materialidade.
* **Validação de Modelos:** Avaliar a robustez matemática e estatística de modelos preditivos.
* **Conformidade Regulatória:** Assegurar que os modelos operam dentro dos limites (Bacen, IFRS 9).
* **Qualidade de Dados:** Auditar a integridade e disponibilidade dos dados em produção.
* **Explicabilidade (XAI):** Garantir transparência em modelos de Machine Learning (XGBoost) via SHAP.
* **Prevenção a Fraude & PLD:** Utilização de métodos estatísticos para detecção de anomalias transacionais.

---

## 📂 Estrutura do Repositório

```text
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
│   ├── 13_backtesting_loss_comparison.py # Comparação Estimado vs. Realizado
│   ├── 14_irrbb_nii_sensitivity.py       # Sensibilidade da Margem Financeira (NII)
│   ├── 15_liquidity_cashflow_stress.py   # Horizonte de Sobrevivência de Caixa
│   └── 16_fraud_detection_benford.py     # Auditoria de PLD (Lei de Benford)
├── tests/                                # Auditoria Automatizada
│   └── test_model_assumptions.py         # Testes rigorosos em Pytest
└── README.md                             # Documentação do projeto

📝 Resumo dos Scripts
🏛️ Governança e Inventário
12_model_inventory_manager.py: Mantém o inventário centralizado, classificando modelos por materialidade e status de validação (MRM).

💳 Risco de Crédito & IFRS 9
01_data_generation.py: Gera dados sintéticos e injeta falhas propositais para testar a resiliência.

02_model_development.py: Compara modelos tradicionais e de ML para Probabilidade de Default.

13_backtesting_loss_comparison.py: Avalia o desempenho real comparando a perda estimada vs. a inadimplência observada.

📈 Risco de Mercado, Liquidez e Monitoramento
06_irrbb_eve_simulation.py: Mede a sensibilidade do patrimônio a choques na curva de juros (EVE).

14_irrbb_nii_sensitivity.py: Audita o impacto de variações de taxas de juros na Margem Financeira (NII).

15_liquidity_cashflow_stress.py: Valida o horizonte de sobrevivência em cenários de estresse severo de fluxo de caixa.

10_market_risk_var_backtesting.py: Valida estatisticamente o modelo de VaR via teste de Kupiec.

🛡️ Prevenção a Fraude e PLD
16_fraud_detection_benford.py: Aplica a Lei de Benford para identificar anomalias estatísticas em volumes transacionais, auxiliando na prevenção à lavagem de dinheiro.

🔍 Explicabilidade e Apresentação
09_model_explainability_shap.py: Valida a lógica econômica das variáveis em modelos de Machine Learning.

11_validation_dashboard_plots.py: Gera o Dashboard visual para reporte executivo e comitês.

⚙️ Instalação e Execução
Instale as dependências:

pip install pandas numpy scikit-learn xgboost scipy joblib shap matplotlib seaborn pytest
Execute a esteira (dentro da pasta src):

cd src
python 01_data_generation.py
python 02_model_development.py
# Execute os demais conforme necessário
Para executar a Auditoria Independente:
A partir da raiz do projeto, rode:

Bash
python -m pytest tests/ -v
👨‍💻 Autor
Gilberto Ricardo Bonatti