# Model Risk Management & Validation Suite

Uma suite abrangente e automatizada em Python para o desenvolvimento, monitorização e validação independente de modelos financeiros (Independent Model Validation - IMV). 

Este repositório serve como um portfólio prático das metodologias exigidas pelas principais normas financeiras e regulatórias (IFRS 9, Basileia III, Bacen), cobrindo múltiplos domínios de risco: Crédito, Mercado, Liquidez e Precificação de Produtos Financeiros.

## 🎯 Objetivos do Projeto

- **Validação de Modelos:** Avaliar a robustez matemática e estatística de modelos preditivos e estruturais, garantindo a ausência de enviesamentos e a estabilidade ao longo do tempo.
- **Conformidade Regulatória (Compliance):** Assegurar que os modelos operam dentro dos limites estabelecidos por normas internas e externas.
- **Qualidade de Dados (Data Quality):** Auditar a integridade, fiabilidade e disponibilidade dos dados utilizados nas fases de desenvolvimento e operação.
- **Testes de Stress & Limites:** Aplicar choques macroeconómicos e testar o comportamento assintótico das equações de precificação.

## 📂 Estrutura do Repositório

```text
model-risk-validation-suite/
│
├── data/                                 # Datasets sintéticos e artefactos (modelos .pkl)
├── src/                                  # Código-fonte principal
│   ├── 01_data_generation.py             # Geração de dados de crédito e testes de Data Quality
│   ├── 02_model_development.py           # Treino de modelos (Regressão Logística e XGBoost)
│   ├── 03_model_validation.py            # Validação estatística (KS, Gini, Calibração)
│   ├── 04_model_monitoring.py            # Monitorização de degradação via PSI (Data Drift)
│   ├── 05_ifrs9_ecl_calculation.py       # Cálculo da Perda Esperada de Crédito (PD, LGD, EAD)
│   ├── 06_irrbb_eve_simulation.py        # Risco de Taxa de Juro e simulação do EVE
│   ├── 07_liquidity_risk_lcr.py          # Validação de Risco de Liquidez (LCR sob stress)
│   └── 08_financial_products_pricing.py  # Precificação de Derivativos (Black-Scholes e Gregas)
│
├── tests/                                # Testes Unitários e Validação de Premissas
│   └── test_model_assumptions.py         # Testes rigorosos em Pytest para auditoria de limites
│
└── README.md                             # Documentação do projeto
