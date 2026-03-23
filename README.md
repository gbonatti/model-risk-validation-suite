# Model Risk Management & Validation Suite

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![pytest](https://img.shields.io/badge/pytest-passing-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Normas](https://img.shields.io/badge/Normas-IFRS%209%20%7C%20Basel%20III%20%7C%20Bacen-orange)

Uma suite abrangente e automatizada em Python para o desenvolvimento, monitorização e validação independente de modelos financeiros (**Independent Model Validation — IMV**).

Este repositório serve como portfólio prático das metodologias exigidas pelas principais normas financeiras e regulatórias (**IFRS 9, Basileia III, Bacen**), cobrindo múltiplos domínios de risco: **Crédito, Mercado, Liquidez** e **Precificação de Produtos Financeiros**.

---

## Sumário

- [Objetivos do Projeto](#-objetivos-do-projeto)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Resumo dos Scripts](#-resumo-dos-scripts)
- [Métodos Estatísticos Utilizados](#-métodos-estatísticos-utilizados)
- [Instalação e Execução](#-instalação-e-execução)
- [Execução da Suite de Testes](#-execução-da-suite-de-testes)
- [Autor](#-autor)

---

## 🎯 Objetivos do Projeto

- **Validação de Modelos:** Avaliar a robustez matemática e estatística de modelos preditivos e estruturais, garantindo a ausência de enviesamentos e a estabilidade ao longo do tempo.
- **Conformidade Regulatória (Compliance):** Assegurar que os modelos operam dentro dos limites estabelecidos por normas internas e externas (IFRS 9, Basileia III, Resolução BCB nº 265/2022).
- **Qualidade de Dados (Data Quality):** Auditar a integridade, fiabilidade e disponibilidade dos dados utilizados nas fases de desenvolvimento e operação.
- **Testes de Stress & Limites:** Aplicar choques macroeconômicos e testar o comportamento assintótico das equações de precificação.
- **Explicabilidade (XAI):** Garantir que modelos complexos (como XGBoost) respeitam a lógica econômica e regulatória através de técnicas como SHAP.

---

## 📂 Estrutura do Repositório

```text
model-risk-validation-suite/
│
├── data/                                 # Artefatos gerados pelo pipeline (ignorados pelo git)
│   └── plots/                            # Gráficos e dashboards (.png)
│
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
│   ├── conftest.py                       # Configuração do pytest (path setup)
│   └── test_model_assumptions.py         # 25+ testes de auditoria por módulo
│
├── .gitignore                            # Exclui artefatos binários e cache
├── requirements.txt                      # Dependências com versões mínimas
├── DOCUMENTATION.md                      # Documentação técnica com equações e métodos
└── README.md                             # Este arquivo
```

---

## 📝 Resumo dos Scripts

### Risco de Crédito & IFRS 9

| Script | Descrição |
|---|---|
| `01_data_generation.py` | Gera um dataset sintético de 50.000 registros de crédito. Calcula a probabilidade de default via função logística e introduz ruídos (nulos, valores sentinela) para testar a resiliência do pipeline. |
| `02_model_development.py` | Treina dois modelos de Probabilidade de Default (PD): **Regressão Logística** (benchmark regulatório) e **XGBoost** (modelo candidato). Ambos os artefatos são serializados em `.pkl`. |
| `03_model_validation.py` | Validação de performance: calcula **Gini Index**, **Curva ROC/AUC** e **KS Statistic** para os dois modelos, comparando com benchmarks regulatórios. |
| `04_model_monitoring.py` | Monitorização contínua usando o **Population Stability Index (PSI)** para capturar drift e instabilidade em novas safras. Emite alertas Verde/Amarelo/Vermelho. |
| `05_ifrs9_ecl_calculation.py` | Calcula a **Perda de Crédito Esperada (ECL)** integrando PD, LGD e EAD (equação fundamental do IFRS 9). Reprodutível via `random_state=42`. |

### Risco de Mercado, Liquidez e Precificação

| Script | Descrição |
|---|---|
| `06_irrbb_eve_simulation.py` | Simula choques paralelos na curva de taxas de juros e audita o impacto no **Valor Econômico do Patrimônio (EVE)** — framework IRRBB, Basileia III. |
| `07_liquidity_risk_lcr.py` | Valida as premissas de escoamento de depósitos (*run-off rates*) sob cenários de stress severo para auditar o indicador **LCR** (mínimo regulatório: 100%). |
| `08_financial_products_pricing.py` | Audita limites matemáticos e sensibilidades (Gregas) do modelo **Black-Scholes**, incluindo guarda para volatilidade nula (caso assintótico). |
| `10_market_risk_var_backtesting.py` | Aplica o **Teste de Kupiec (POF)** para garantir que as exceções históricas do modelo de **VaR 99%** estão dentro do limite estatístico aceitável. |

### Explicabilidade, Auditoria e Apresentação

| Script | Descrição |
|---|---|
| `09_model_explainability_shap.py` | Desconstrói o XGBoost usando **valores SHAP** para validar a monotonicidade das variáveis — garante que "juros altos = risco alto" é aprendido corretamente. |
| `11_validation_dashboard_plots.py` | Consolida as métricas críticas em um **Dashboard Visual** (4 painéis): ROC/Gini, Calibração por Decil, Série Temporal de PSI e Cenários IRRBB. |
| `tests/test_model_assumptions.py` | Suite de **25+ testes automatizados** em pytest: valida axiomas matemáticos, condições de limite e restrições normativas, importando as funções reais de produção. |

---

## 📐 Métodos Estatísticos Utilizados

| Método | Script | Finalidade |
|---|---|---|
| Distribuição Log-Normal | `01` | Modelagem de renda e valor de empréstimo |
| Regressão Logística (Sigmoide) | `01`, `02` | Geração da target e benchmark regulatório de PD |
| XGBoost / Gradient Boosting | `02` | Modelo candidato de PD |
| AUC-ROC & Índice de Gini | `03`, `11` | Discriminação do modelo de crédito |
| Estatística KS (Kolmogorov-Smirnov) | `03` | Separação entre bons e maus pagadores |
| PSI (Population Stability Index) | `04`, `11` | Monitoramento de drift e estabilidade |
| ECL = PD × LGD × EAD | `05` | Provisão de crédito esperada (IFRS 9) |
| Valor Presente / Desconto | `06` | Precificação de fluxos de caixa (EVE) |
| LCR com run-off rates | `07` | Liquidez sob stress (Basileia III) |
| Black-Scholes & Delta | `08` | Precificação de opções europeias |
| Valores de Shapley (SHAP) | `09` | Explicabilidade e auditoria de monotonicidade |
| Teste de Kupiec (LR χ²) | `10` | Backtesting de VaR |

> Para a documentação completa com equações LaTeX, derivações e benchmarks regulatórios, consulte [`DOCUMENTATION.md`](DOCUMENTATION.md).

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
python 04_model_monitoring.py
python 05_ifrs9_ecl_calculation.py
python 06_irrbb_eve_simulation.py
python 07_liquidity_risk_lcr.py
python 08_financial_products_pricing.py
python 09_model_explainability_shap.py
python 10_market_risk_var_backtesting.py
python 11_validation_dashboard_plots.py
```

> **Nota:** Os scripts `01` e `02` devem ser executados primeiro, pois geram os artefatos `.parquet` e `.pkl` usados pelos demais.

---

## 🧪 Execução da Suite de Testes

A partir da **raiz do projeto**, execute:

```bash
pytest tests/ -v
```

Os testes são independentes do pipeline (não requerem os `.pkl` gerados) e validam as funções matemáticas puras de cada módulo. Cobertura atual: **6 domínios, 25+ casos de teste**.

---

## 👨‍💻 Autor

**Gilberto Ricardo Bonatti**
Analista Sênior de Validação de Modelos e Risco

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-blue?logo=linkedin)](https://www.linkedin.com/in/gilberto-bonatti)
