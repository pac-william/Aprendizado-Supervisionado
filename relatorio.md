# Análise Comparativa de Modelos de Aprendizado Supervisionado: Boston Housing Dataset

## 1. Introdução ao Problema
O problema em questão envolve a predição de preços de imóveis no dataset Boston Housing, um conjunto de dados clássico para problemas de regressão. O objetivo principal é comparar diferentes abordagens de aprendizado supervisionado, focando não apenas na precisão, mas principalmente na compreensão do comportamento dos modelos e na justificativa das decisões tomadas.

### 1.1 Contexto
- Dataset histórico usado extensivamente em estudos de regressão
- Problema real de predição de preços de imóveis
- Oportunidade de comparar diferentes abordagens de machine learning

### 1.2 Objetivos
- Implementar e comparar diferentes modelos de regressão
- Analisar o comportamento e interpretabilidade dos modelos
- Justificar as decisões tomadas em cada etapa
- Extrair insights valiosos sobre o problema

## 2. Descrição do Dataset
O dataset Boston Housing contém informações sobre diversas características de imóveis em Boston, incluindo:
- CRIM: Taxa de criminalidade per capita
- ZN: Proporção de terrenos residenciais
- INDUS: Proporção de negócios não-varejistas
- CHAS: Variável dummy para proximidade ao rio Charles
- NOX: Concentração de óxidos de nitrogênio
- RM: Número médio de quartos
- AGE: Proporção de unidades ocupadas construídas antes de 1940
- DIS: Distância ponderada aos centros de emprego
- RAD: Índice de acessibilidade às rodovias
- TAX: Taxa de imposto sobre propriedade
- PTRATIO: Razão aluno-professor
- LSTAT: Status socioeconômico da população
- MEDV: Valor mediano das casas (variável alvo)

### 2.1 Características do Dataset
- 506 observações
- 13 variáveis preditoras
- 1 variável alvo (MEDV)
- Mistura de variáveis numéricas e categóricas
- Presença de valores ausentes

## 3. Metodologia

### 3.1 Abordagem Geral
1. **Análise Exploratória**
   - Visualização dos dados
   - Análise estatística descritiva
   - Identificação de padrões e correlações

2. **Pré-processamento**
   - Tratamento de valores ausentes
   - Normalização
   - Remoção de outliers
   - Engenharia de características

3. **Modelagem**
   - Implementação dos modelos
   - Otimização de hiperparâmetros
   - Validação cruzada
   - Avaliação de performance

### 3.2 Ferramentas Utilizadas
- Python 3.x
- scikit-learn para implementação dos modelos
- pandas para manipulação de dados
- matplotlib e seaborn para visualizações
- numpy para operações numéricas

## 4. EDA e Preparação dos Dados

### 4.1 Análise Exploratória
- Visualização das distribuições das variáveis
- Detecção de outliers através de boxplots
- Análise de correlações entre variáveis
- Identificação de padrões e tendências nos dados

#### 4.1.1 Insights da Análise Exploratória
1. **Distribuições:**
   - Variáveis como CRIM e LSTAT apresentam distribuição assimétrica
   - RM (número de quartos) mostra distribuição mais normal
   - CHAS é binária, indicando proximidade ao rio

2. **Correlações:**
   - Forte correlação negativa entre LSTAT e MEDV
   - Correlação positiva entre RM e MEDV
   - Relações complexas entre variáveis ambientais (NOX, DIS)

### 4.2 Pré-processamento
- Tratamento de valores ausentes usando mediana
- Normalização das variáveis com StandardScaler
- Remoção de outliers usando Z-score
- Engenharia de características:
  - Criação de interações entre variáveis relevantes
  - Seleção de características baseada em correlação

#### 4.2.1 Justificativa das Decisões
1. **Tratamento de Valores Ausentes:**
   - Mediana escolhida por ser robusta a outliers
   - Mantém a distribuição original dos dados

2. **Normalização:**
   - Necessária para modelos sensíveis à escala
   - Melhora a convergência da RNA

3. **Remoção de Outliers:**
   - Z-score > 3 como critério
   - Balanceamento entre remoção de ruído e perda de informação

## 5. Descrição dos Modelos Implementados

### 5.1 Random Forest
- Implementação usando scikit-learn
- Arquitetura: Ensemble de árvores de decisão
- Hiperparâmetros otimizados:
  - Número de estimadores: 200
  - Profundidade máxima: 15
  - Amostras mínimas para split: 5
  - Amostras mínimas por folha: 2

#### 5.1.1 Vantagens do Random Forest
- Robustez a overfitting
- Captura de relações não-lineares
- Importância das características
- Facilidade de interpretação

### 5.2 Rede Neural Artificial
- Implementação usando scikit-learn (MLPRegressor)
- Arquitetura: 3 camadas ocultas (128, 64, 32 neurônios)
- Hiperparâmetros otimizados:
  - Learning rate: 0.01
  - Alpha: 0.0001
  - Early stopping ativado

#### 5.2.1 Vantagens da RNA
- Capacidade de aprendizado complexo
- Adaptabilidade a diferentes padrões
- Potencial para melhor performance

## 6. Resultados e Comparação entre Modelos

### 6.1 Métricas de Performance
**Random Forest:**
- MAE: 2.70
- RMSE: 5.31
- R²: 0.67

**Rede Neural Artificial:**
- MAE: 3.85
- RMSE: 6.13
- R²: 0.55

### 6.2 Análise Comparativa
- **Desempenho:** Random Forest apresentou melhor performance geral
- **Interpretabilidade:** RF mais interpretável, permitindo análise de importância das características
- **Tempo de Treinamento:** RF mais rápido e eficiente
- **Desafios:** Diferentes trade-offs entre complexidade e performance

#### 6.2.1 Análise Crítica dos Resultados
1. **Random Forest:**
   - Performance superior em todas as métricas
   - Maior estabilidade nas predições
   - Melhor interpretabilidade
   - Menor tempo de treinamento

2. **Rede Neural Artificial:**
   - Performance inferior, mas ainda aceitável
   - Maior variabilidade nas predições
   - Difícil interpretação
   - Tempo de treinamento maior

## 7. Conclusões Finais

### 7.1 Aprendizados do Grupo
1. A importância do pré-processamento adequado dos dados
2. O trade-off entre interpretabilidade e performance
3. A necessidade de justificar escolhas de modelos e parâmetros
4. A relevância da análise crítica dos resultados

### 7.2 Insights Principais
1. O Random Forest se mostrou mais adequado para este problema específico
2. A engenharia de características teve impacto significativo nos resultados
3. A interpretabilidade do modelo é crucial para aplicações práticas
4. O balanceamento entre complexidade e performance é essencial

### 7.3 Reflexões Críticas
1. **Sobre a Escolha dos Modelos:**
   - RF foi mais adequado para este problema específico
   - RNA poderia ter melhor performance com mais dados
   - Importância de considerar o contexto do problema

2. **Sobre o Pré-processamento:**
   - Impacto significativo nos resultados
   - Necessidade de justificar cada decisão
   - Balanceamento entre limpeza e perda de informação

3. **Sobre a Interpretabilidade:**
   - Crucial para aplicações práticas
   - Trade-off com performance
   - Necessidade de explicar decisões do modelo

## 8. Observações

### 8.1 Limitações do Estudo
- Tamanho limitado do dataset
- Possibilidade de overfitting em modelos mais complexos
- Dependência da qualidade do pré-processamento
- Limitações computacionais

### 8.2 Sugestões para Trabalhos Futuros
1. Explorar outros algoritmos de ensemble
2. Testar diferentes técnicas de engenharia de características
3. Investigar métodos de interpretação mais avançados
4. Aplicar os modelos em datasets similares para validação
5. Implementar técnicas de regularização mais sofisticadas

### 8.3 Considerações Finais
O foco do trabalho foi na compreensão e interpretação dos modelos, não apenas na maximização da precisão. As decisões tomadas em cada etapa foram justificadas e documentadas, permitindo uma análise crítica dos resultados obtidos. O estudo demonstrou a importância de considerar múltiplos aspectos além da performance pura, como interpretabilidade, tempo de treinamento e robustez do modelo. 