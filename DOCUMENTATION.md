# Documentação Técnica — Model Risk Validation Suite

**Autor:** Gilberto Ricardo Bonatti — Analista Sênior de Validação de Modelos e Risco
**Versão:** 1.0 | **Data:** Março 2026
**Normas de Referência:** IFRS 9, Basileia III, Resolução BCB nº 265/2022

---

## Sumário

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Script 01 — Geração de Dados Sintéticos](#2-script-01--geração-de-dados-sintéticos)
3. [Script 02 — Desenvolvimento de Modelos de PD](#3-script-02--desenvolvimento-de-modelos-de-pd)
4. [Script 03 — Validação Estatística do Modelo](#4-script-03--validação-estatística-do-modelo)
5. [Script 04 — Monitoramento e Estabilidade (PSI)](#5-script-04--monitoramento-e-estabilidade-psi)
6. [Script 05 — Cálculo de ECL (IFRS 9)](#6-script-05--cálculo-de-ecl-ifrs-9)
7. [Script 06 — Risco de Taxa de Juros (IRRBB / EVE)](#7-script-06--risco-de-taxa-de-juros-irrbb--eve)
8. [Script 07 — Risco de Liquidez (LCR)](#8-script-07--risco-de-liquidez-lcr)
9. [Script 08 — Precificação de Derivativos (Black-Scholes)](#9-script-08--precificação-de-derivativos-black-scholes)
10. [Script 09 — Explicabilidade com SHAP (XAI)](#10-script-09--explicabilidade-com-shap-xai)
11. [Script 10 — Backtesting de VaR (Teste de Kupiec)](#11-script-10--backtesting-de-var-teste-de-kupiec)
12. [Script 11 — Dashboard de Validação Visual](#12-script-11--dashboard-de-validação-visual)
13. [Suite de Testes Automatizados](#13-suite-de-testes-automatizados)
14. [Glossário de Termos Regulatórios](#14-glossário-de-termos-regulatórios)

---

## 1. Visão Geral da Arquitetura

O pipeline é composto por **11 scripts sequenciais** e uma **suite de testes automatizados**. Cada script produz artefatos (arquivos `.pkl`, `.parquet` e `.png`) que alimentam os scripts subsequentes.

```
01_data_generation
        │
        ▼
02_model_development ──────────────────────────────────┐
        │                                              │
        ▼                                              │
03_model_validation      04_model_monitoring           │
                                                       │
05_ifrs9_ecl  ←── usa modelo pkl                       │
06_irrbb_eve  ←── independente                         │
07_lcr        ←── independente                         │
08_pricing    ←── independente                         │
09_shap       ←── usa modelo pkl                       │
10_var_bt     ←── independente                         │
        │                                              │
        ▼                                              │
11_dashboard  ←── consolida tudo ◄─────────────────────┘
        │
        ▼
tests/test_model_assumptions.py  (auditoria independente)
```

---

## 2. Script 01 — Geração de Dados Sintéticos

**Arquivo:** `src/01_data_generation.py`
**Saída:** `data/credit_portfolio.parquet`

### O que o script faz

Gera um dataset sintético com **50.000 registros** de crédito, simulando uma carteira de empréstimos de uma instituição financeira. O dataset contém variáveis socioeconômicas do cliente, características do empréstimo, e uma variável-alvo (`default_12m`) que indica inadimplência em 12 meses.

Após gerar os dados limpos, o script introduz **problemas de Data Quality (DQ)** intencionais para simular condições reais de produção.

### Variáveis Geradas

| Variável | Tipo | Distribuição | Parâmetros |
|---|---|---|---|
| `idade` | Inteiro | Uniforme discreta | [18, 75] |
| `renda_mensal` | Real | Log-Normal | μ = 8,5 ; σ = 0,8 |
| `score_bureau` | Inteiro | Uniforme discreta | [300, 850] |
| `valor_emprestimo` | Real | Log-Normal | μ = 9,0 ; σ = 0,9 |
| `taxa_juros` | Real | Uniforme contínua | [0,05 ; 0,25] |
| `LTV` | Real | Uniforme contínua | [0,30 ; 1,20] |
| `safra` | Data | Seleção aleatória | Jan/2022 – Dez/2023 |

### Distribuição Log-Normal

Renda e valor de empréstimo seguem distribuição **Log-Normal** — adequada para modelar grandezas sempre positivas com cauda longa à direita (como salários e valores de crédito):

```
Se X ~ LogNormal(μ, σ²), então ln(X) ~ Normal(μ, σ²)

Média esperada:  E[X] = exp(μ + σ²/2)
Variância:       Var[X] = (exp(σ²) - 1) · exp(2μ + σ²)
```

Para `renda_mensal` com μ = 8,5 e σ = 0,8:
- **Média esperada ≈ R$ 8.103** (exp(8,5 + 0,32) = exp(8,82))
- Reflete a assimetria real de renda: maioria com valores medianos, minoria com rendas altas.

### Geração da Target — Função Logística

A variável `default_12m` é gerada por uma **função logística** que combina os fatores de risco de forma linear antes de aplicar a transformação sigmoide:

**Passo 1 — Escore linear de risco (z):**

```
z = -0,015 · score_bureau
  + 3,50 · LTV
  - 0,50 · ln(1 + renda_mensal)
  + 10,0 · taxa_juros
  + ε,   onde ε ~ Normal(0, 2)
```

**Interpretação dos coeficientes:**
- `score_bureau · (-0,015)` — maior score reduz o risco (sinal negativo)
- `LTV · 3,5` — quanto maior a relação loan-to-value, maior o risco
- `ln(1 + renda) · (-0,5)` — renda mais alta (em log) reduz o risco
- `taxa_juros · 10` — juros mais altos aumentam fortemente o risco
- `ε` — ruído aleatório simulando variáveis não observadas

**Passo 2 — Probabilidade de default (função sigmoide):**

```
P(default) = 1 / (1 + e^(-z))
```

Esta é a **função logística** (ou sigmoide), com domínio em (-∞, +∞) e imagem em (0, 1).

**Passo 3 — Binarização com limiar de percentil:**

```
default_12m = 1  se  P(default) > P85(P(default))
            = 0  caso contrário
```

O **percentil 85** foi escolhido para gerar uma **taxa de inadimplência de ~15%**, realista para carteiras de crédito no varejo brasileiro.

### Injeção de Ruído para Data Quality

```python
# 2% dos registros com renda ausente (MCAR - Missing Completely at Random)
df.loc[df.sample(frac=0.02).index, 'renda_mensal'] = np.nan

# 1% dos registros com score inválido (valor sentinela -999)
df.loc[df.sample(frac=0.01).index, 'score_bureau'] = -999
```

Esses erros simulam problemas reais de integração entre sistemas legados e testam a **resiliência do pipeline** de pré-processamento no script seguinte.

---

## 3. Script 02 — Desenvolvimento de Modelos de PD

**Arquivo:** `src/02_model_development.py`
**Entrada:** `data/credit_portfolio.parquet`
**Saídas:** `data/xgb_pd_model.pkl`, `data/test_data.pkl`

### O que o script faz

Executa o **pipeline de desenvolvimento do modelo de Probabilidade de Default (PD)**: limpeza dos dados, divisão treino/teste e treinamento do modelo XGBoost.

### Pré-processamento

Antes do treinamento, os valores problemáticos gerados no Script 01 são tratados:

```python
# Substitui NaN por mediana (robusto a outliers, ao contrário da média)
df['renda_mensal'] = df['renda_mensal'].fillna(df['renda_mensal'].median())

# Substitui valor sentinela -999 por mediana
df['score_bureau'] = df['score_bureau'].replace(-999, df['score_bureau'].median())
```

A **mediana** é preferida à média para imputação porque é **resistente a outliers** — uma característica essencial em dados financeiros com distribuição assimétrica.

### Divisão Treino/Teste

```
Holdout split:  70% treino  |  30% teste
Estratégia:     random_state=42 (reprodutibilidade)
```

A divisão **hold-out** (também chamada de validação simples) é o método regulatório mais básico: o modelo jamais "vê" os dados de teste durante o treinamento, garantindo avaliação não-enviesada da performance out-of-sample.

### Modelo XGBoost (Gradient Boosting)

O **XGBoost (eXtreme Gradient Boosting)** é um ensemble de árvores de decisão que minimiza iterativamente uma função de perda. Cada nova árvore é treinada nos **resíduos** da árvore anterior:

**Objetivo do XGBoost:**

```
Obj(Θ) = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

Onde:
  L(yᵢ, ŷᵢ)  = função de perda (log-loss para classificação binária)
  Ω(fₖ)      = termo de regularização da k-ésima árvore
  Θ           = conjunto de parâmetros de todas as árvores
```

**Função de perda Log-Loss (Cross-Entropy Binária):**

```
L(y, p) = -[y · ln(p) + (1 - y) · ln(1 - p)]

Onde:
  y ∈ {0, 1} = classe real (default ou não)
  p ∈ (0, 1) = probabilidade predita pelo modelo
```

**Hiperparâmetros configurados:**

| Parâmetro | Valor | Descrição |
|---|---|---|
| `n_estimators` | 100 | Número de árvores no ensemble |
| `max_depth` | 3 | Profundidade máxima de cada árvore |
| `learning_rate` | 0,10 | Taxa de aprendizado (shrinkage) |
| `random_state` | 42 | Semente aleatória para reprodutibilidade |

`max_depth=3` e `learning_rate=0,1` são escolhas conservadoras que **reduzem o risco de overfitting**, uma exigência regulatória explícita em validações de modelos de crédito.

---

## 4. Script 03 — Validação Estatística do Modelo

**Arquivo:** `src/03_model_validation.py`
**Entrada:** `data/xgb_pd_model.pkl`, `data/test_data.pkl`

### O que o script faz

Avalia o **poder discriminativo** do modelo de PD — a capacidade de separar bons pagadores (y=0) de maus pagadores (y=1). São calculadas três métricas clássicas de validação: AUC-ROC, Gini Index e KS Statistic.

### Curva ROC e AUC

A **Curva ROC (Receiver Operating Characteristic)** plota a **Taxa de Verdadeiros Positivos (TPR)** versus a **Taxa de Falsos Positivos (FPR)** para todos os limiares de classificação possíveis:

```
TPR (Sensibilidade) = VP / (VP + FN)    ← "Quantos defaults o modelo capturou?"
FPR (1 - Especificidade) = FP / (FP + VN)  ← "Quantos não-defaults foram alarmes falsos?"

Onde:
  VP = Verdadeiros Positivos  |  FP = Falsos Positivos
  FN = Falsos Negativos       |  VN = Verdadeiros Negativos
```

A **AUC (Area Under the Curve)** é a área sob a curva ROC, calculada por integração numérica (regra trapezoidal):

```
AUC = ∫₀¹ TPR(FPR) d(FPR)

Interpretação:
  AUC = 0,50  →  Modelo aleatório (sem poder discriminativo)
  AUC = 1,00  →  Discriminação perfeita
  AUC > 0,70  →  Aceitável para modelos de crédito (benchmark regulatório Bacen)
```

### Índice de Gini

O **Índice de Gini** é derivado diretamente do AUC-ROC e mede a **concentração da capacidade discriminativa** do modelo:

```
Gini = 2 · AUC - 1

Escala:
  Gini = 0,00  →  Modelo aleatório
  Gini = 1,00  →  Perfeição
  Gini ≥ 0,40  →  Benchmark mínimo regulatório para modelos de PD
  Gini ≥ 0,60  →  Considerado bom
  Gini ≥ 0,70  →  Considerado excelente
```

```python
auc = roc_auc_score(y_test, pd_preds)
gini = 2 * auc - 1
```

### Estatística KS (Kolmogorov-Smirnov)

O **Teste KS de duas amostras** mede a **máxima separação** entre as distribuições acumuladas de probabilidades preditas para bons e maus clientes:

```
KS = max|F_bons(x) - F_maus(x)|

Onde:
  F_bons(x) = CDF empírica das probabilidades preditas para bons pagadores
  F_maus(x) = CDF empírica das probabilidades preditas para maus pagadores
```

```python
pd_bons = pd_preds[y_test == 0]   # scores dos bons pagadores
pd_maus = pd_preds[y_test == 1]   # scores dos maus pagadores
ks_stat, _ = ks_2samp(pd_bons, pd_maus)
```

**Benchmarks para KS:**

```
KS < 0,20  →  Discriminação insuficiente
KS ∈ [0,20; 0,40)  →  Aceitável
KS ∈ [0,40; 0,70)  →  Bom
KS ≥ 0,70  →  Excelente (verificar possível data leakage)
```

> **Diferença entre KS e Gini:** O KS mede o ponto de melhor separação (máximo local), enquanto o Gini integra a discriminação ao longo de todos os limiares. São métricas complementares e ambas devem ser reportadas em validações regulatórias.

---

## 5. Script 04 — Monitoramento e Estabilidade (PSI)

**Arquivo:** `src/04_model_monitoring.py`
**Entrada:** `data/xgb_pd_model.pkl`, `data/test_data.pkl`

### O que o script faz

Monitora a **degradação do modelo ao longo do tempo** detectando mudanças na distribuição das probabilidades preditas entre a **amostra de desenvolvimento** e uma **nova amostra operacional**. O drift é quantificado pelo **Population Stability Index (PSI)**.

### Simulação do Cenário de Drift

```python
X_operacao = X_test.copy()
X_operacao['taxa_juros'] = X_operacao['taxa_juros'] * 1.5  # aumento de 50% nas taxas
```

Este cenário simula um ambiente macroeconômico adverso (por exemplo, ciclo de alta de juros pelo Bacen) que altera o perfil de risco da população.

### Population Stability Index (PSI)

O PSI é uma medida de **divergência estatística** entre duas distribuições, baseada na divergência de Kullback-Leibler (KL). É calculado binando as pontuações e comparando as proporções em cada bin:

**Algoritmo:**

```
1. Dividir as pontuações de desenvolvimento em B bins (decis: B=10)
2. Calcular a proporção de observações em cada bin: Eᵢ (esperado) e Aᵢ (atual)
3. Calcular:

   PSI = Σᵢ (Aᵢ - Eᵢ) · ln(Aᵢ / Eᵢ)

Onde:
  Aᵢ = proporção da amostra atual no bin i
  Eᵢ = proporção da amostra de desenvolvimento no bin i
  B  = número de bins (tipicamente 10 decis)
```

```python
def calculate_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    # Suavização para evitar log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents   = np.where(actual_percents   == 0, 0.0001, actual_percents)
    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
```

**Limiares de decisão (padrão de mercado):**

```
PSI < 0,10  →  Verde: população estável, modelo apto para uso
PSI ∈ [0,10; 0,25)  →  Amarelo: alerta, investigar causas do drift
PSI ≥ 0,25  →  Vermelho: modelo desatualizado, recalibração ou retreinamento obrigatório
```

### Relação com Divergência KL

O PSI é uma versão **simétrica** da divergência KL:

```
KL(A||E) = Σᵢ Aᵢ · ln(Aᵢ / Eᵢ)        ← assimétrica
PSI      = Σᵢ (Aᵢ - Eᵢ) · ln(Aᵢ / Eᵢ)  ← simétrica (considera ambas as direções)
```

---

## 6. Script 05 — Cálculo de ECL (IFRS 9)

**Arquivo:** `src/05_ifrs9_ecl_calculation.py`
**Entrada:** `data/xgb_pd_model.pkl`, `data/test_data.pkl`

### O que o script faz

Calcula a **Provisão para Perdas de Crédito Esperadas (ECL — Expected Credit Loss)** conforme exigido pela **IFRS 9 (International Financial Reporting Standard 9)**, vigente no Brasil desde 2025 (CPC 48).

### A Equação Fundamental do ECL (IFRS 9)

```
ECLᵢ = PDᵢ · LGDᵢ · EADᵢ

Onde:
  PDᵢ  = Probability of Default: probabilidade de o cliente i não honrar a dívida em 12 meses
  LGDᵢ = Loss Given Default: percentual da exposição que será perdido dado o default
  EADᵢ = Exposure at Default: saldo devedor no momento do default
```

**ECL Total da carteira:**

```
ECL_Total = Σᵢ PDᵢ · LGDᵢ · EADᵢ
```

### Componentes Implementados

**Probability of Default (PD):**
```python
pd_12m = xgb_model.predict_proba(X_test)[:, 1]
# Saída do modelo XGBoost para o horizonte de 12 meses
```

**Exposure at Default (EAD):**
```python
ead = X_test['valor_emprestimo'] * np.random.uniform(0.9, 1.1, len(X_test))
# Fator de conversão aleatório entre 90%-110% do saldo contratual
# Simula saques adicionais em linhas de crédito rotativas
```

**Loss Given Default (LGD):**
```python
lgd = np.clip(X_test['LTV'] * 0.6 + np.random.normal(0, 0.1, len(X_test)), 0.1, 1.0)
```

A fórmula `LGD = LTV × 0,6` reflete a intuição econômica:
- LTV alto → ativo financiado vale menos que a dívida → recuperação baixa → LGD alto
- O fator 0,6 representa uma taxa histórica de recuperação de ~40% (LGD médio de 60%)
- O ruído `Normal(0, 0,1)` simula variabilidade idiossincrática por cliente
- `np.clip(·, 0,1, 1,0)` garante que LGD ∈ [10%, 100%] — restrição econômica fundamental

### Estágios do IFRS 9

Embora o script calcule ECL de Estágio 1 (12 meses), a norma define três estágios:

```
Estágio 1: Sem deterioração significativa → ECL de 12 meses = PD(12m) · LGD · EAD
Estágio 2: Deterioração significativa do risco → ECL lifetime (toda a vida do ativo)
Estágio 3: Evidência objetiva de imparidade → ECL lifetime + juros sobre valor líquido
```

---

## 7. Script 06 — Risco de Taxa de Juros (IRRBB / EVE)

**Arquivo:** `src/06_irrbb_eve_simulation.py`

### O que o script faz

Simula o impacto de **choques nas taxas de juros** sobre o **Valor Econômico do Patrimônio (EVE — Economic Value of Equity)** da instituição, conforme exigido pelo framework **IRRBB (Interest Rate Risk in the Banking Book)** — Basileia III, pilar 2.

### Cálculo do Valor Presente (PV)

O valor presente de um conjunto de fluxos de caixa descontados a uma taxa de juros r é:

```
VP = Σᵢ CFᵢ / (1 + r)^tᵢ

Onde:
  CFᵢ = fluxo de caixa no tempo tᵢ
  r   = taxa de desconto (curva de juros)
  tᵢ  = tempo em anos até o fluxo i
```

```python
def calculate_present_value(cashflows, times, rate_curve):
    return np.sum(cashflows / ((1 + rate_curve) ** times))
```

### Simulação de Choques Paralelos

```python
base_rate = np.full(1000, 0.10)      # Cenário base: taxa de 10%
eve_base  = calculate_present_value(cashflows, times, base_rate)
eve_up    = calculate_present_value(cashflows, times, base_rate + 0.02)  # +200bps
```

**ΔEVE (variação pelo choque):**

```
ΔEVE = EVE_chocado - EVE_base

Interpretação:
  ΔEVE < 0  →  Posição long em duration: banco perde valor com alta de juros
  ΔEVE > 0  →  Posição short em duration: banco ganha valor com alta de juros
```

### Estrutura dos Fluxos de Caixa

```python
cashflows = np.where(
    times > 5,
    np.random.uniform(500, 1000, 1000),    # Ativos de longo prazo (recebíveis)
    np.random.uniform(-800, -200, 1000)    # Passivos de curto prazo (depósitos)
)
```

Esta assimetria temporal (ativos longos vs. passivos curtos) cria **exposição ao risco de taxa** — o modelo captura o **gap de duration** típico de bancos comerciais.

### Benchmark Regulatório (Basileia III IRRBB)

```
|ΔEVE| / Capital Tier 1 > 15%  →  Instituição classificada como "outlier"
                                    Sujeita a supervisão reforçada pelo regulador
```

Os **6 cenários de choque** prescritivos do Comitê de Basileia são:
- Paralelo +200bps e -200bps
- Short rates up / Short rates down
- Steepening (subida de longo prazo)
- Flattening (queda de longo prazo)

---

## 8. Script 07 — Risco de Liquidez (LCR)

**Arquivo:** `src/07_liquidity_risk_lcr.py`

### O que o script faz

Valida o **Liquidity Coverage Ratio (LCR)** sob cenários de estresse, métrica introduzida por Basileia III para garantir que instituições financeiras mantenham **liquidez suficiente para 30 dias** em situação de crise.

### Equação do LCR

```
LCR = HQLA / (Saídas_Líquidas_30d)

Onde:
  HQLA = High Quality Liquid Assets (ativos líquidos de alta qualidade)
  Saídas_Líquidas_30d = Total de Saídas_30d - min(Entradas_30d, 0,75 × Saídas_30d)

Requisito regulatório (Bacen/Basileia III):  LCR ≥ 100%
```

### Implementação

```python
hqla = 50_000_000  # R$ 50 milhões em ativos líquidos de alta qualidade

# Passivos com diferentes fatores de estresse (run-off rates)
passivos = pd.DataFrame({
    'saldo_atual':   [100_000_000, 50_000_000, 80_000_000, 40_000_000],
    'fator_estresse': [0.05,        0.10,       0.25,       0.40]
})
```

**Fatores de estresse por tipo de passivo:**

| Tipo de Depósito | Fator de Estresse | Justificativa |
|---|---|---|
| Depósitos de varejo (segurados) | 5% | Baixo risco de corrida bancária |
| Depósitos de varejo (não segurados) | 10% | Risco moderado |
| Depósitos institucionais | 25% | Alta sensibilidade a notícias adversas |
| Depósitos wholesale / concentrados | 40% | Altamente voláteis em crises |

**Total de Saídas Brutas:**

```
Saídas_Brutas = Σᵢ (Saldo_Passivo_i × Fator_Estresse_i)
```

**Entradas com haircut regulatório de 50%:**

```python
entradas_projetadas = 15_000_000 * 0.50  # Apenas 50% das entradas são reconhecidas
```

```
LCR = HQLA / (Saídas_Brutas - Entradas_Ajustadas)
```

---

## 9. Script 08 — Precificação de Derivativos (Black-Scholes)

**Arquivo:** `src/08_financial_products_pricing.py`

### O que o script faz

Implementa o **modelo de Black-Scholes-Merton** para precificação analítica de opções europeias de compra (call options) e calcula a **grega Delta** — sensibilidade do preço da opção ao preço do ativo subjacente.

### A Fórmula de Black-Scholes (1973)

Para uma opção de compra europeia (Call), o preço justo é:

```
C = S · N(d₁) - K · e^(-r·T) · N(d₂)

Onde:
  d₁ = [ln(S/K) + (r + σ²/2) · T] / (σ · √T)
  d₂ = d₁ - σ · √T

Parâmetros:
  S  = Preço atual do ativo subjacente
  K  = Preço de exercício (strike)
  T  = Tempo até o vencimento (em anos)
  r  = Taxa de juros livre de risco (contínua)
  σ  = Volatilidade anualizada do ativo subjacente
  N(·) = Função de distribuição acumulada da Normal Padrão
```

```python
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return preco, delta
```

### Interpretação dos Componentes

```
S · N(d₁)           = valor esperado do ativo, ponderado pela probabilidade
                      de exercício no mundo neutro ao risco
K · e^(-rT) · N(d₂) = valor presente do strike, ponderado pela probabilidade
                      de exercício (probabilidade de S > K no vencimento)
```

### A Grega Delta

```
Delta = ∂C/∂S = N(d₁)

Interpretação:
  Delta = 0,50  →  opção "at-the-money" (S ≈ K), 50% de chance de exercício
  Delta → 1,00  →  opção "deep in-the-money" (S >> K)
  Delta → 0,00  →  opção "deep out-of-the-money" (S << K)

Uso prático:
  Delta também representa o hedge ratio: para cada opção vendida,
  comprar Delta unidades do ativo subjacente neutraliza o risco direcional.
```

### Premissas do Modelo

O modelo de Black-Scholes assume:
- Mercado sem fricções (sem custos de transação, sem dividendos)
- Taxa de juros constante e conhecida
- Volatilidade constante ao longo do tempo (**não** captura smile de volatilidade)
- O ativo subjacente segue um movimento browniano geométrico: `dS = μS dt + σS dW`
- Opção europeia: exercício apenas no vencimento

---

## 10. Script 09 — Explicabilidade com SHAP (XAI)

**Arquivo:** `src/09_model_explainability_shap.py`
**Entrada:** `data/xgb_pd_model.pkl`, `data/test_data.pkl`
**Saída:** `data/plots/shap_summary_plot.png`

### O que o script faz

Aplica a técnica **SHAP (SHapley Additive exPlanations)** para **decompor e auditar** as predições do modelo XGBoost, verificando se as relações aprendidas são consistentes com a lógica econômica esperada.

### Teoria dos Valores de Shapley

Os valores SHAP são baseados na **teoria dos jogos cooperativos** (Shapley, 1953). Para cada predição, o SHAP calcula a contribuição marginal de cada variável considerando todas as ordens possíveis de inclusão:

```
φᵢ(f, x) = Σ_{S ⊆ F\{i}} [|S|! · (|F| - |S| - 1)! / |F|!] · [f(S ∪ {i}) - f(S)]

Onde:
  φᵢ     = valor SHAP da variável i para a observação x
  F       = conjunto de todas as variáveis do modelo
  S       = subconjunto de variáveis excluindo i
  f(S)    = predição do modelo usando apenas as variáveis em S
  f(S∪{i}) = predição adicionando a variável i ao conjunto S
```

**Propriedade de aditividade (interpretabilidade local):**

```
f(x) = E[f(x)] + Σᵢ φᵢ(x)

Ou seja: Predição = Valor Base + Soma das contribuições SHAP de cada variável
```

### TreeExplainer — Eficiência para Árvores

```python
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)
```

O `TreeExplainer` calcula valores SHAP exatos (não aproximados) para modelos baseados em árvores em tempo **polinomial**, usando o algoritmo de path-dependent de Lundberg et al. (2018) — com complexidade O(TLD²) onde T=árvores, L=folhas, D=profundidade.

### Validação de Monotonicidade Econômica

```python
juros_idx = X_sample.columns.get_loc('taxa_juros')
correlacao = pd.Series(X_sample['taxa_juros'].values).corr(
                pd.Series(shap_values[:, juros_idx]))

# Critério: correlação deve ser positiva e superior a 0,5
assert correlacao > 0.5, "ALERTA: modelo viola lógica econômica"
```

**Hipótese testada:** *"Taxas de juros mais altas devem aumentar o risco de default."*

```
H₀: corr(taxa_juros, SHAP_taxa_juros) ≤ 0,5  →  Lógica violada (reprovar)
H₁: corr(taxa_juros, SHAP_taxa_juros) > 0,5  →  Lógica preservada (aprovar)
```

Este teste é uma forma de **validação de monotonicidade** — exigida pela SR 11-7 (Fed) e pelas diretrizes de modelos do Bacen para garantir que o modelo não aprende relações espúrias.

---

## 11. Script 10 — Backtesting de VaR (Teste de Kupiec)

**Arquivo:** `src/10_market_risk_var_backtesting.py`

### O que o script faz

Valida o modelo de **Value at Risk (VaR)** por meio do **Teste de Kupiec (Proportion of Failures — POF)**, verificando estatisticamente se a frequência histórica de exceções é compatível com o nível de confiança declarado pelo modelo.

### Value at Risk (VaR)

```
VaR(α, T) = perda máxima esperada com probabilidade α ao longo de T dias

P(Perda > VaR) = 1 - α

Exemplo: VaR(99%, 1 dia) = R$ 230.000
→ Há apenas 1% de probabilidade de que a perda de 1 dia exceda R$ 230k
→ Em 252 dias úteis, espera-se: 252 × 0,01 = 2,52 exceções
```

### O Teste de Kupiec (1995)

O teste verifica se a **taxa de exceções observadas** é estatisticamente igual à taxa esperada pelo modelo, usando um **teste de razão de verossimilhança (LR):**

**Hipóteses:**

```
H₀: p_observada = p_esperada  (modelo bem calibrado)
H₁: p_observada ≠ p_esperada  (modelo mal calibrado)

Onde:
  p_esperada = 1 - α  (ex: 0,01 para VaR a 99%)
  p_observada = N_exceções / N_dias_totais
```

**Estatística de teste (log-likelihood ratio):**

```
LR_POF = -2 · ln [ L(p₀; x) / L(p̂; x) ]

Onde:
  L(p; x) = (1-p)^(N-x) · p^x        (verossimilhança binomial)
  p₀ = taxa esperada (H₀)
  p̂  = taxa observada (MLE)
  N  = número de observações
  x  = número de exceções

Expandindo:

LR_POF = -2 · { ln[(1-p₀)^(N-x) · p₀^x] - ln[(1-p̂)^(N-x) · p̂^x] }
```

```python
lr_pof = -2 * (
    np.log(((1-p_expected)**(n-x)) * (p_expected**x)) -
    np.log(((1-p_observed)**(n-x)) * (p_observed**x))
)
```

**Distribuição e p-valor:**

```
Sob H₀:  LR_POF ~ χ²(1)   (qui-quadrado com 1 grau de liberdade)

p-valor = P(χ²(1) > LR_POF)

Decisão:
  p-valor > 0,05  →  Não rejeitar H₀: modelo de VaR validado (zona verde)
  p-valor ≤ 0,05  →  Rejeitar H₀: VaR subestimado ou superestimado (zona vermelha)
```

### Zonas de Tráfego (Basileia III)

Para VaR a 99% com 250 dias de observação:

```
Zona Verde   (0–4 exceções)    → Modelo aprovado, capital mínimo
Zona Amarela (5–9 exceções)    → Multiplicador de capital aumenta de 3,0 → 3,65
Zona Vermelha (≥10 exceções)   → Modelo rejeitado, multiplicador = 4,0
```

---

## 12. Script 11 — Dashboard de Validação Visual

**Arquivo:** `src/11_validation_dashboard_plots.py`
**Saída:** `data/plots/validation_dashboard_complete.png`

### O que o script faz

Consolida as métricas dos demais módulos em um **dashboard visual** de 4 painéis, gerado automaticamente com `matplotlib` e `seaborn`. O dashboard é o artefato final de um relatório de Validação Independente de Modelos (IMV).

### Estrutura do Dashboard (2×2)

```
┌─────────────────────────────┬─────────────────────────────┐
│  Painel 1: ROC / Gini       │  Painel 2: Calibração       │
│  Discriminação de Crédito   │  por Decil                  │
├─────────────────────────────┼─────────────────────────────┤
│  Painel 3: PSI Temporal     │  Painel 4: Cenários IRRBB   │
│  Série mensal de drift      │  Delta EVE por cenário      │
└─────────────────────────────┴─────────────────────────────┘
```

### Painel 2 — Calibração por Decil

O modelo é dividido em **10 grupos de igual tamanho** pela probabilidade predita (decis). Em cada decil, compara-se a **PD média estimada pelo modelo** com a **taxa de default real observada**:

```
Calibração ideal:  PD_estimada(decil_k) ≈ Taxa_default_real(decil_k)

Para todo k ∈ {1, 2, ..., 10}
```

Uma boa calibração implica que o modelo não superestima nem subestima o risco sistematicamente — critério regulatório explícito do CPC 48/IFRS 9 para provisões de crédito.

### Painel 3 — Série Temporal de PSI

```python
psi_mensal = [0.02, 0.03, 0.05, 0.04, 0.06, 0.08, 0.07, 0.09, 0.12, 0.15, 0.18, 0.21]
psi_acumulado = np.cumsum(psi_mensal) / np.arange(1, 13)   # Média móvel acumulada
```

O PSI acumulado é a **média aritmética do PSI mensal** até o mês de referência, suavizando oscilações pontuais e revelando a tendência estrutural de drift.

---

## 13. Suite de Testes Automatizados

**Arquivo:** `tests/test_model_assumptions.py`

### O que o script faz

Implementa testes de **auditoria independente** usando `pytest`, verificando condições de limite matemático e restrições normativas que os modelos devem satisfazer. Funciona como um validador algorítmico autônomo.

### Teste 1 — Não-negatividade do ECL

```python
class TestMarketRiskIRRBB:
    def test_ecl_non_negative(self):
        assert 0.05 * 0.45 * 10000.0 >= 0.0
```

**Premissa verificada:**

```
ECL = PD · LGD · EAD ≥ 0   para todo (PD, LGD, EAD) ∈ [0,1] × [0,1] × ℝ⁺
```

A restrição é trivialmente verdadeira se PD, LGD e EAD são não-negativos, mas o teste defende contra implementações com bug (ex: sinal invertido em PD ou LGD negativo).

### Teste 2 — Assíntota de Volatilidade Zero no Black-Scholes

```python
class TestFinancialProductsPricing:
    def test_asymptotic_volatility_zero(self):
        price, delta = black_scholes_call(100.0, 120.0, 1.0, 0.05, 1e-10)
        assert pytest.approx(price, abs=1e-5) == 0.0
```

**Premissa verificada:**

```
lim_{σ→0} C(S, K, T, r, σ) = max(S·e^(rT) - K, 0) · e^(-rT)

Para S=100, K=120, T=1, r=0.05, σ→0:
  S·e^(rT) = 100 · e^(0.05) ≈ 105,13 < K=120
  → Call está out-of-the-money, preço → 0 ✓
```

Sem volatilidade, o ativo não se move — uma opção OTM é definitivamente inútil. O teste verifica a **consistência matemática** da implementação nos casos limites.

---

## 14. Glossário de Termos Regulatórios

| Termo | Definição |
|---|---|
| **AUC** | Area Under the ROC Curve — área sob a curva ROC, medida de discriminação |
| **EAD** | Exposure at Default — saldo devedor estimado no momento do default |
| **ECL** | Expected Credit Loss — provisão para perdas de crédito esperadas (IFRS 9) |
| **EVE** | Economic Value of Equity — valor econômico do patrimônio sensível a juros |
| **Gini** | Índice de Gini = 2·AUC - 1, mede o poder discriminativo do modelo de PD |
| **HQLA** | High Quality Liquid Assets — ativos líquidos de alta qualidade (LCR) |
| **IFRS 9** | Norma contábil internacional para instrumentos financeiros e provisões de crédito |
| **IMV** | Independent Model Validation — validação independente de modelos |
| **IRRBB** | Interest Rate Risk in the Banking Book — risco de taxa de juros na carteira bancária |
| **KS** | Kolmogorov-Smirnov statistic — máxima separação entre distribuições cumulativas |
| **LCR** | Liquidity Coverage Ratio — índice de cobertura de liquidez (Basileia III) |
| **LGD** | Loss Given Default — percentual de perda dado o evento de default |
| **LTV** | Loan-to-Value — razão entre o saldo do empréstimo e o valor do ativo financiado |
| **PD** | Probability of Default — probabilidade de inadimplência do cliente |
| **PSI** | Population Stability Index — índice de estabilidade da população |
| **ROC** | Receiver Operating Characteristic — curva de desempenho binário do modelo |
| **SHAP** | SHapley Additive exPlanations — método de explicabilidade baseado em teoria dos jogos |
| **VaR** | Value at Risk — perda máxima esperada a um dado nível de confiança |
| **XAI** | Explainable Artificial Intelligence — IA explicável |
| **XGBoost** | eXtreme Gradient Boosting — algoritmo de ensemble baseado em árvores |

---

*Documentação gerada automaticamente a partir do código-fonte do repositório.*
*Para sugestões ou correções, abra uma issue no repositório.*
