# Relatório de Pré-processamento - Boston Housing Dataset

## 1. Tratamento de Valores Ausentes

### Análise Inicial
- O dataset Boston Housing possui valores ausentes em várias colunas
- Colunas com valores ausentes: CRIM, ZN, INDUS, CHAS, AGE, LSTAT
- Total de valores ausentes por coluna:
  - CRIM: 20 valores
  - ZN: 20 valores
  - INDUS: 20 valores
  - CHAS: 20 valores
  - AGE: 20 valores
  - LSTAT: 20 valores

### Estratégia de Tratamento
- Decidimos preencher os valores ausentes com a mediana de cada coluna
- Justificativa: A mediana é menos sensível a outliers que a média
- Após o tratamento, não há mais valores ausentes no dataset

## 2. Normalização das Variáveis

### Método Utilizado
- Utilizamos o StandardScaler do scikit-learn
- Transformação: z = (x - μ) / σ
- Onde:
  - x: valor original
  - μ: média da variável
  - σ: desvio padrão da variável

### Resultados
- Todas as variáveis preditoras foram normalizadas
- Após a normalização:
  - Média ≈ 0
  - Desvio Padrão ≈ 1
- Isso ajuda a melhorar a performance dos modelos de machine learning

## 3. Remoção de Outliers

### Método Utilizado
- Cálculo do Z-score para cada variável
- Remoção de linhas com Z-score > 3 em qualquer variável
- Z-score = (x - μ) / σ

### Resultados
- Número de linhas com outliers: 103
- Porcentagem de outliers: 20.36%
- Shape do dataset após remoção: (403, 13)

## 4. Análise de Correlações

### Método Utilizado
- Cálculo da matriz de correlação de Pearson
- Visualização através de heatmap
- Gráfico salvo como 'correlacao_pos_preprocessamento.png'

### Observações
- [Será preenchido após execução com as principais correlações encontradas]

## 5. Conclusões

### Impacto do Pré-processamento
- Melhoria na qualidade dos dados
- Remoção de ruído
- Normalização para melhor performance dos modelos

### Próximos Passos
- Implementação dos modelos de machine learning
- Avaliação da performance dos modelos
- Comparação entre diferentes algoritmos 