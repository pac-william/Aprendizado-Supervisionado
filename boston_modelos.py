import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def selecionar_e_engenhar_caracteristicas(X, y):
    """
    Seleciona e engenha características
    
    Parâmetros:
    -----------
    X : DataFrame
        Variáveis preditoras
    y : Series
        Variável alvo
        
    Retorna:
    --------
    X_selected : DataFrame
        Dados com características selecionadas e engenhadas
    """
    # 1. Seleção de características baseada em correlação
    print("\n1. Selecionando características baseadas em correlação...")
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    print("Correlação com a variável alvo:")
    print(corr_with_target)
    
    # Selecionar características com correlação > 0.1
    selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
    print(f"Características selecionadas: {selected_features}")
    
    # 2. Engenharia de características
    print("\n2. Engenhando novas características...")
    X_selected = X[selected_features].copy()
    
    # Exemplo: Criar interação entre RM e LSTAT
    X_selected['RM_LSTAT'] = X['RM'] * X['LSTAT']
    
    # Exemplo: Criar interação entre NOX e DIS
    X_selected['NOX_DIS'] = X['NOX'] * X['DIS']
    
    print("Novas características criadas: RM_LSTAT, NOX_DIS")
    
    return X_selected

def criar_modelo_rna(input_dim):
    """
    Cria uma rede neural artificial usando MLPRegressor com arquitetura mais robusta
    """
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),  # Mais camadas e neurônios
        max_iter=2000,                     # Mais iterações
        early_stopping=True,               # Early stopping
        validation_fraction=0.1,           # 10% dos dados para validação
        n_iter_no_change=10,              # Número de iterações sem melhoria
        random_state=42
    )

def criar_modelo_rf():
    """
    Cria um Random Forest com parâmetros otimizados
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

def otimizar_hiperparametros(modelo, X_train, y_train, param_grid):
    """
    Otimiza os hiperparâmetros do modelo usando GridSearchCV
    """
    grid_search = GridSearchCV(
        modelo,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def avaliar_modelo_cv(modelo, X, y, cv=5):
    """
    Avalia o modelo usando validação cruzada
    """
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"RMSE médio (CV): {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")
    return rmse_scores

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

def visualizar_distribuicoes(df):
    """
    Cria visualizações das distribuições das variáveis
    """
    # Criar subplots para cada variável
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        # Histograma
        sns.histplot(data=df, x=col, ax=axes[idx], kde=True)
        axes[idx].set_title(f'Distribuição de {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequência')
    
    # Remover subplots vazios
    for idx in range(len(df.columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('distribuicoes_variaveis.png')
    plt.close()

def visualizar_boxplots(df):
    """
    Cria boxplots para detectar outliers
    """
    # Criar subplots para cada variável
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        # Boxplot
        sns.boxplot(data=df, y=col, ax=axes[idx])
        axes[idx].set_title(f'Boxplot de {col}')
        axes[idx].set_ylabel(col)
    
    # Remover subplots vazios
    for idx in range(len(df.columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('boxplots_variaveis.png')
    plt.close()

def visualizar_correlacoes(df):
    """
    Cria matriz de correlação com heatmap
    """
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('matriz_correlacao.png')
    plt.close()

def visualizar_predicoes(y_true, y_pred_rna, y_pred_rf):
    """
    Cria gráficos comparando valores reais e preditos
    """
    # Gráfico de dispersão para RNA
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred_rna, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('RNA: Valores Reais vs Preditos')
    plt.tight_layout()
    plt.savefig('rna_predicoes.png')
    plt.close()
    
    # Gráfico de dispersão para RF
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred_rf, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('RF: Valores Reais vs Preditos')
    plt.tight_layout()
    plt.savefig('rf_predicoes.png')
    plt.close()

if __name__ == "__main__":
    # Carregar dados
    df = carregar_dados()
    
    if df is not None:
        # Visualizações exploratórias
        print("\nCriando visualizações exploratórias...")
        visualizar_distribuicoes(df)
        visualizar_boxplots(df)
        visualizar_correlacoes(df)
        
        # Separar variáveis preditoras e alvo
        X = df.drop('MEDV', axis=1)
        y = df['MEDV']
        
        # Pré-processar dados
        X_clean, y_clean, scaler = preprocessar_dados(X, y)
        
        # Selecionar e engenhar características
        X_selected = selecionar_e_engenhar_caracteristicas(X, y)
        
        # Remover linhas com valores NaN
        X_selected = X_selected.dropna()
        y = y[X_selected.index]
        
        # Separar dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        # Definir grids de hiperparâmetros para cada modelo
        rna_param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
        
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Treinar e otimizar modelos
        print("\nTreinando e otimizando modelos...")
        
        # 1. Rede Neural Artificial (RNA)
        print("\n1. Treinando e otimizando Rede Neural Artificial (RNA)...")
        modelo_rna = criar_modelo_rna(X_train.shape[1])
        modelo_rna_otimizado = otimizar_hiperparametros(modelo_rna, X_train, y_train, rna_param_grid)
        print("Melhores parâmetros RNA:", modelo_rna_otimizado.get_params())
        
        # Avaliar RNA com validação cruzada
        print("\nAvaliando RNA com validação cruzada:")
        scores_rna = avaliar_modelo_cv(modelo_rna_otimizado, X_train, y_train)
        
        # Treinar RNA final com todos os dados de treino
        modelo_rna_otimizado.fit(X_train, y_train)
        y_pred_rna = modelo_rna_otimizado.predict(X_test)
        
        print("\nResultados finais da RNA:")
        avaliar_modelo(y_test, y_pred_rna)
        
        # 2. Random Forest (RF)
        print("\n2. Treinando e otimizando Random Forest (RF)...")
        modelo_rf = criar_modelo_rf()
        modelo_rf_otimizado = otimizar_hiperparametros(modelo_rf, X_train, y_train, rf_param_grid)
        print("Melhores parâmetros RF:", modelo_rf_otimizado.get_params())
        
        # Avaliar RF com validação cruzada
        print("\nAvaliando RF com validação cruzada:")
        scores_rf = avaliar_modelo_cv(modelo_rf_otimizado, X_train, y_train)
        
        # Treinar RF final com todos os dados de treino
        modelo_rf_otimizado.fit(X_train, y_train)
        y_pred_rf = modelo_rf_otimizado.predict(X_test)
        
        print("\nResultados finais da RF:")
        avaliar_modelo(y_test, y_pred_rf)
        
        # Visualizar predições
        print("\nCriando visualizações das predições...")
        visualizar_predicoes(y_test, y_pred_rna, y_pred_rf)
        
        # Atualizar relatório com resultados
        num_outliers = (X.shape[0] - X_clean.shape[0])
        porcentagem_outliers = (num_outliers / X.shape[0]) * 100
        atualizar_relatorio(num_outliers, porcentagem_outliers, X_clean.shape)
        
        print("\nPré-processamento, seleção de características e treinamento dos modelos concluídos com sucesso!")
        print("Relatório atualizado com os resultados.")
