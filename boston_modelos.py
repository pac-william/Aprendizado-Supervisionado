import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Configurações de visualização
plt.style.use('seaborn')
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
    """
    # Tratar valores ausentes
    X = X.fillna(X.median())
    
    # Normalizar variáveis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Remover ruído (opcional: remover outliers)
    # Aqui, vamos apenas remover linhas com valores extremos (exemplo: Z-score > 3)
    z_scores = np.abs((X_scaled - X_scaled.mean()) / X_scaled.std())
    X_clean = X_scaled[(z_scores < 3).all(axis=1)]
    y_clean = y[(z_scores < 3).all(axis=1)]
    
    return X_clean, y_clean

def criar_modelo_rna(input_dim):
    """
    Cria uma rede neural artificial
    """
    modelo = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return modelo

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

if __name__ == "__main__":
    # TODO: Implementar fluxo principal
    pass
