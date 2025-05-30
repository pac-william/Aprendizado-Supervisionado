# Projeto de Aprendizado Supervisionado - Boston Housing

Este projeto implementa diferentes modelos de aprendizado supervisionado para prever preços de casas usando o dataset Boston Housing.

## Estrutura do Projeto

- `boston_modelos.py`: Implementação dos modelos (RNA, Árvore de Decisão e Random Forest)
- `boston_eda.py`: Análise exploratória dos dados
- `boston_preprocessamento.py`: Limpeza e preparação dos dados
- `requirements.txt`: Dependências do projeto

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Dataset

O dataset Boston Housing contém informações sobre casas em Boston, incluindo:
- Preço médio das casas (variável alvo)
- Características como taxa de criminalidade, proporção de áreas residenciais, etc.

## Modelos Implementados

1. Rede Neural Artificial (RNA)
2. Árvore de Decisão (DT)
3. Random Forest (RF)

## Métricas de Avaliação

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coeficiente de Determinação) 