# Model Risk Validation Suite

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![pytest](https://img.shields.io/badge/pytest-passing-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Normas](https://img.shields.io/badge/Normas-IFRS%209%20%7C%20Basel%20III%20%7C%20Bacen-orange)

Uma suíte técnica em Python para o desenvolvimento, monitoramento e validação independente de modelos financeiros (**Independent Model Validation — IMV**).

Este repositório reúne implementações práticas de metodologias exigidas pelas normas financeiras (**IFRS 9, Basileia III, Bacen**), cobrindo riscos de Crédito, Mercado, Liquidez e Auditoria de Dados.

---

## Sumário

- [Escopo do Projeto](#-escopo-do-projeto)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Resumo das Atividades Técnicas](#-resumo-das-atividades-técnicas)
- [Métodos Estatísticos Utilizados](#-métodos-estatísticos-utilizados)
- [Instalação e Execução](#-instalação-e-execução)
- [Execução da Suite de Testes](#-execução-da-suite-de-testes)
- [Autor](#-autor)

---

## 🎯 Escopo do Projeto

- **Execução de Validação:** Aplicar testes estatísticos para medir a robustez de modelos preditivos.
- **Monitoramento de Performance:** Acompanhar a estabilidade dos modelos (PSI) e realizar backtesting periódico.
- **Data Quality:** Auditar a integridade das bases de dados utilizadas em modelagem.
- **Explicabilidade (XAI):** Utilizar técnicas de SHAP para interpretar as saídas de modelos de Machine Learning.
- **Stress Testing:** Simular impactos de cenários macroeconômicos em indicadores de capital e liquidez.

---

## 📂 Estrutura do Repositório

```text
model-risk-validation-suite/
│
├── data/                                 # Artefatos gerados pelo pipeline (ignorados pelo git)
│   └── plots/                            # Gráficos e dashboards (.png)
│
├── src/                                  # Scripts de execução técnica
│   ├── 01_data_generation.py             # Preparação de dados e testes de DQ
│   ├── 02_model_development.py           # Treinamento de modelos (Logística e XGBoost)
│   ├── 03_model_validation.py            # Testes de discriminação (KS, Gini)
│   ├── 04_model_monitoring.py            # Cálculo de estabilidade populacional (PSI)
│   ├── 05_ifrs9_ecl_calculation.py       # Cálculo de Perda Esperada (ECL — IFRS 9)
│   ├── 06_irrbb_eve_simulation.py        # Simulação de risco de taxa de juro (EVE)
│   ├── 07_liquidity_risk_lcr.py          # Monitoramento de liquidez (LCR)
│   ├── 08_financial_products_pricing.py  # Auditoria de precificação (Black-Scholes)
│   ├── 09_model_explainability_shap.py   # Interpretação de modelos com SHAP
│   ├── 10_market_risk_var_backtesting.py # Backtesting de VaR (Kupiec)
│   ├── 11_validation_dashboard_plots.py  # Visualização de métricas técnicas
│   ├── 12_model_inventory_manager.py     # Organização do inventário de modelos
│   ├── 13_backtesting_loss_comparison.py # Comparação entre perda estimada vs. observada
│   ├── 14_irrbb_nii_sensitivity.py       # Sensibilidade da margem financeira (NII)
│   ├── 15_liquidity_cashflow_stress.py   # Teste de estresse de fluxo de caixa
│   └── 16_fraud_detection_benford.py     # Detecção de anomalias estatísticas (PLD)
│
├── tests/                                # Auditoria Automatizada
│   ├── conftest.py                       # Configuração do pytest (path setup)
│   └── test_model_assumptions.py         # 25+ testes de auditoria por módulo
│
├── .gitignore                            # Exclui artefatos binários e cache
├── requirements.txt                      # Dependências com versões mínimas
├── DOCUMENTATION.md                      # Documentação técnica com equações e métodos
└── README.md                             # Este arquivo
```

---

## 📝 Resumo das Atividades Técnicas

### 🛠️ Modelagem & Crédito

| Scripts | Descrição |
|---|---|
| `01`, `02`, `03` | Geração de safras, tratamento de nulos, treinamento de algoritmos (**Regressão Logística** + **XGBoost**) e validação de poder preditivo (Gini/KS). |
| `05`, `13` | Execução do cálculo de **ECL (IFRS 9)** e backtesting para verificar a aderência das perdas estimadas vs. observadas. |

### 📊 Mercado, Liquidez & Monitoramento

| Scripts | Descrição |
|---|---|
| `04`, `06`, `14` | Monitoramento de data drift via **PSI** e análise de sensibilidade do balanço a choques de juros (**EVE** e **NII** — IRRBB). |
| `07`, `15` | Acompanhamento do indicador **LCR** e simulação de sobrevivência de caixa em cenários de estresse severo. |
| `10` | Validação estatística de modelos de **VaR** para risco de mercado (Teste de Kupiec). |

### 🔍 Auditoria e Interpretação

| Scripts | Descrição |
|---|---|
| `09` | Tradução das variáveis do XGBoost via **SHAP** para garantir que o modelo segue premissas econômicas (monotonicidade). |
| `16` | Aplicação da **Lei de Benford** para identificar desvios em bases transacionais (PLD/AML). |
| `11` | Dashboard visual consolidado: ROC/Gini, Calibração por Decil, PSI Temporal e Cenários IRRBB. |
| `tests/` | Rotinas de **25+ testes automatizados** para garantir que os modelos respeitam limites normativos. |

---

## 📐 Métodos Estatísticos Utilizados

| Método | Script(s) | Finalidade |
|---|---|---|
| Distribuição Log-Normal | `01` | Modelagem de renda e valor de empréstimo |
| Regressão Logística (Sigmoide) | `01`, `02` | Geração da target e benchmark regulatório de PD |
| XGBoost / Gradient Boosting | `02` | Modelo candidato de PD |
| AUC-ROC & Índice de Gini | `03`, `11` | Discriminação do modelo de crédito |
| Estatística KS (Kolmogorov-Smirnov) | `03` | Separação entre bons e maus pagadores |
| PSI (Population Stability Index) | `04`, `11` | Monitoramento de drift e estabilidade |
| ECL = PD × LGD × EAD | `05`, `13` | Provisão de crédito esperada (IFRS 9) |
| Valor Presente / Desconto | `06`, `14` | Precificação de fluxos de caixa (EVE/NII) |
| LCR com run-off rates | `07`, `15` | Liquidez sob stress (Basileia III) |
| Black-Scholes & Delta | `08` | Precificação de opções europeias |
| Valores de Shapley (SHAP) | `09` | Explicabilidade e auditoria de monotonicidade |
| Teste de Kupiec (LR χ²) | `10` | Backtesting de VaR |
| Lei de Benford | `16` | Detecção de anomalias em dados transacionais |

> Para a documentação completa com equações, derivações e benchmarks regulatórios, consulte [`DOCUMENTATION.md`](DOCUMENTATION.md).

---

## ⚙️ Instalação e Execução

Recomenda-se o uso de um ambiente virtual Python (versão **3.8+**).

**1. Clone o repositório e crie o ambiente:**

```bash
git clone https://github.com/gbonatti/model-risk-validation-suite.git
cd model-risk-validation-suite
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# ou: .venv\Scripts\activate     # Windows
```

**2. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**3. Execute o pipeline sequencialmente a partir da pasta `src/`:**

```bash
cd src
python 01_data_generation.py
python 02_model_development.py
python 03_model_validation.py
# ... e assim por diante até o script 16
```

> **Nota:** Os scripts `01` e `02` devem ser executados primeiro, pois geram os artefatos `.parquet` e `.pkl` usados pelos demais.

---

## 🧪 Execução da Suite de Testes

A partir da **raiz do projeto**, execute:

```bash
pytest tests/ -v
```

Os testes são independentes do pipeline (não requerem os `.pkl` gerados) e validam as funções matemáticas puras de cada módulo. Cobertura: **6 domínios, 25+ casos de teste**.

---

## 👨‍💻 Autor (Uso do Claude pra otmização)

**Gilberto Ricardo Bonatti** - Especialista em Modelagem Numérica

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-blue?logo=linkedin)](https://www.linkedin.com/in/gilberto-bonatti)
