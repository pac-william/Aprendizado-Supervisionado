import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de visualização
sns.set_palette("husl")

def carregar_dados(caminho_arquivo='HousingData.csv'):
    """
    Carrega o dataset Boston Housing
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"Dataset carregado com sucesso!")
        print(f"Shape do dataset: {df.shape}")
        print("\nPrimeiras linhas do dataset:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None

def analise_estatistica(df):
    """
    Realiza análise estatística básica dos dados
    """
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    print("\nInformações sobre valores nulos:")
    print(df.isnull().sum())

def plotar_distribuicoes(df):
    """
    Plota as distribuições das variáveis
    """
    # Criando subplots para cada variável
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        sns.histplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title(f'Distribuição de {col}')
    
    # Removendo subplots vazios
    for idx in range(len(df.columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def plotar_correlacoes(df):
    """
    Plota a matriz de correlação
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()

def detectar_outliers(df):
    """
    Detecta outliers nas variáveis
    """
    # Criando boxplots para cada variável
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        sns.boxplot(data=df, y=col, ax=axes[idx])
        axes[idx].set_title(f'Boxplot de {col}')
    
    # Removendo subplots vazios
    for idx in range(len(df.columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carregando os dados
    df = carregar_dados()
    
    if df is not None:
        # Realizando análise estatística
        analise_estatistica(df)
        
        # Plotando distribuições
        plotar_distribuicoes(df)
        
        # Plotando correlações
        plotar_correlacoes(df)
        
        # Detectando outliers
        detectar_outliers(df) 