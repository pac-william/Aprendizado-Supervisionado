import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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
        return df
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None

def preprocessar_dados(X, y):
    """
    Realiza o pré-processamento dos dados
    
    Parâmetros:
    -----------
    X : DataFrame
        Variáveis preditoras
    y : Series
        Variável alvo
        
    Retorna:
    --------
    X_clean : array
        Dados preprocessados das variáveis preditoras
    y_clean : array
        Dados preprocessados da variável alvo
    scaler : StandardScaler
        Objeto scaler para normalização
    """
    # 1. Tratar valores ausentes
    print("\n1. Tratando valores ausentes...")
    print("Valores ausentes antes do tratamento:")
    print(X.isnull().sum())
    
    # Preencher valores ausentes com a mediana
    X = X.fillna(X.median())
    
    print("\nValores ausentes depois do tratamento:")
    print(X.isnull().sum())
    
    # 2. Normalizar variáveis
    print("\n2. Normalizando variáveis...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Criar DataFrame com dados normalizados para análise
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("\nEstatísticas após normalização:")
    print(X_scaled_df.describe())
    
    # 3. Remover ruído (outliers)
    print("\n3. Removendo outliers...")
    # Calcular Z-scores
    z_scores = np.abs((X_scaled - X_scaled.mean()) / X_scaled.std())
    
    # Identificar linhas com outliers (Z-score > 3)
    outliers = (z_scores > 3).any(axis=1)
    print(f"Número de linhas com outliers: {outliers.sum()}")
    print(f"Porcentagem de outliers: {(outliers.sum() / len(X)) * 100:.2f}%")
    
    # Remover linhas com outliers
    X_clean = X_scaled[~outliers]
    y_clean = y[~outliers]
    
    print(f"\nShape após remoção de outliers: {X_clean.shape}")
    
    # 4. Verificar correlações após pré-processamento
    print("\n4. Verificando correlações após pré-processamento...")
    X_clean_df = pd.DataFrame(X_clean, columns=X.columns)
    corr_matrix = X_clean_df.corr()
    
    # Plotar matriz de correlação
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação após Pré-processamento')
    plt.tight_layout()
    plt.savefig('correlacao_pos_preprocessamento.png')
    plt.close()
    
    return X_clean, y_clean, scaler

def criar_modelo_rna(input_dim):
    """
    Cria uma rede neural artificial usando MLPRegressor
    """
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

def criar_modelo_dt():
    """
    Cria uma árvore de decisão
    """
    return DecisionTreeRegressor(random_state=42)

def criar_modelo_rf():
    """
    Cria um Random Forest
    """
    return RandomForestRegressor(n_estimators=100, random_state=42)

def avaliar_modelo(y_true, y_pred):
    """
    Avalia o modelo usando métricas de regressão
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    return mae, rmse, r2

def atualizar_relatorio(num_outliers, porcentagem_outliers, shape_final):
    """
    Atualiza o relatório com os resultados do pré-processamento
    """
    with open('relatorio.md', 'r', encoding='utf-8') as file:
        conteudo = file.read()
    
    # Atualizar resultados da remoção de outliers
    conteudo = conteudo.replace(
        "Número de linhas com outliers: [será preenchido após execução]",
        f"Número de linhas com outliers: {num_outliers}"
    )
    conteudo = conteudo.replace(
        "Porcentagem de outliers: [será preenchido após execução]",
        f"Porcentagem de outliers: {porcentagem_outliers:.2f}%"
    )
    conteudo = conteudo.replace(
        "Shape do dataset após remoção: [será preenchido após execução]",
        f"Shape do dataset após remoção: {shape_final}"
    )
    
    with open('relatorio.md', 'w', encoding='utf-8') as file:
        file.write(conteudo)

if __name__ == "__main__":
    # Carregar dados
    df = carregar_dados()
    
    if df is not None:
        # Separar variáveis preditoras e alvo
        X = df.drop('MEDV', axis=1)  # Todas as colunas exceto MEDV
        y = df['MEDV']  # Coluna MEDV (preço das casas)
        
        # Pré-processar dados
        X_clean, y_clean, scaler = preprocessar_dados(X, y)
        
        # Atualizar relatório com resultados
        num_outliers = (X.shape[0] - X_clean.shape[0])
        porcentagem_outliers = (num_outliers / X.shape[0]) * 100
        atualizar_relatorio(num_outliers, porcentagem_outliers, X_clean.shape)
        
        print("\nPré-processamento concluído com sucesso!")
        print("Relatório atualizado com os resultados.")
