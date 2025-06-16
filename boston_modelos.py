"""
Script para análise preditiva do dataset Boston Housing usando Redes Neurais e Random Forest.
Realiza pré-processamento, seleção de features, treinamento e avaliação de modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
import os
import json

# Configurações de visualização
sns.set_palette("husl")

def criar_pasta_resultados():
    """Cria pasta para armazenar resultados e visualizações"""
    if not os.path.exists('result'):
        os.makedirs('result')
        print("Pasta 'result' criada com sucesso!")

def carregar_dados(caminho_arquivo='HousingData.csv'):
    """Carrega e retorna o dataset Boston Housing"""
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
    Realiza pré-processamento dos dados:
    1. Tratamento de valores ausentes
    2. Normalização das variáveis
    3. Remoção de outliers
    4. Análise de correlações
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
    plt.savefig('result/correlacao_pos_preprocessamento.png')
    plt.close()
    
    return X_clean, y_clean, scaler

def selecionar_e_engenhar_caracteristicas(X, y):
    """
    Seleciona features relevantes e cria novas features:
    1. Seleção baseada em correlação com target
    2. Criação de features de interação
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
    """Configura e retorna uma Rede Neural com arquitetura otimizada"""
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),  # Mais camadas e neurônios
        max_iter=2000,                     # Mais iterações
        early_stopping=True,               # Early stopping
        validation_fraction=0.1,           # 10% dos dados para validação
        n_iter_no_change=10,              # Número de iterações sem melhoria
        random_state=42
    )

def criar_modelo_rf():
    """Configura e retorna um Random Forest com parâmetros otimizados"""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

def otimizar_hiperparametros(modelo, X_train, y_train, param_grid):
    """Otimiza hiperparâmetros do modelo usando GridSearchCV"""
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
    """Avalia modelo usando validação cruzada e retorna scores RMSE"""
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"RMSE médio (CV): {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")
    return rmse_scores

def analisar_residuos(y_true, y_pred, modelo_nome):
    """
    Analisa resíduos do modelo:
    1. Gráfico de dispersão
    2. Histograma
    3. QQ-plot
    4. Resíduos padronizados
    """
    residuos = y_true - y_pred
    residuos_padronizados = residuos / np.std(residuos)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Dispersão: Resíduos vs. Valores Preditos
    ax = axes[0, 0]
    ax.scatter(y_pred, residuos, alpha=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set(xlabel='Valores Preditos', ylabel='Resíduos',
           title=f'{modelo_nome}: Resíduos vs Valores Preditos')

    # 2. Histograma dos resíduos
    ax = axes[0, 1]
    sns.histplot(residuos, kde=True, ax=ax)
    ax.set(xlabel='Resíduos', ylabel='Frequência',
           title=f'{modelo_nome}: Distribuição dos Resíduos')

    # 3. QQ-plot
    ax = axes[1, 0]
    stats.probplot(residuos, dist="norm", plot=ax)
    ax.set_title(f'{modelo_nome}: QQ-Plot dos Resíduos')

    # 4. Dispersão: Resíduos Padronizados vs. Valores Preditos
    ax = axes[1, 1]
    ax.scatter(y_pred, residuos_padronizados, alpha=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.axhline(2, color='g', linestyle='--')
    ax.axhline(-2, color='g', linestyle='--')
    ax.set(xlabel='Valores Preditos', ylabel='Resíduos Padronizados',
           title=f'{modelo_nome}: Resíduos Padronizados vs Valores Preditos')

    plt.tight_layout()
    plt.savefig(f'result/analise_residuos_{modelo_nome.lower()}.png')
    plt.close()

    # Estatísticas descritivas
    media = np.mean(residuos)
    desvio = np.std(residuos)
    assimetria = stats.skew(residuos)
    curtose = stats.kurtosis(residuos)
    _, p_valor = stats.normaltest(residuos)

    print(f"\nAnálise de Resíduos - {modelo_nome}:")
    print(f"Média dos resíduos: {media:.4f}")
    print(f"Desvio padrão dos resíduos: {desvio:.4f}")
    print(f"Skewness dos resíduos: {assimetria:.4f}")
    print(f"Kurtosis dos resíduos: {curtose:.4f}")
    print(f"Teste de normalidade (p-valor): {p_valor:.4f}")

    return residuos
def analisar_importancia_features(modelo, X, modelo_nome):
    """Analisa e visualiza importância das features para modelos que suportam feature_importances_"""
    if hasattr(modelo, 'feature_importances_'):
        # Obter importância das features
        importancia = modelo.feature_importances_
        
        # Criar DataFrame com as importâncias
        df_importancia = pd.DataFrame({
            'Feature': X.columns,
            'Importância': importancia
        })
        
        # Ordenar por importância
        df_importancia = df_importancia.sort_values('Importância', ascending=False)
        
        # Plotar gráfico de barras
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_importancia, x='Importância', y='Feature')
        plt.title(f'{modelo_nome}: Importância das Features')
        plt.tight_layout()
        plt.savefig(f'result/importancia_features_{modelo_nome.lower()}.png')
        plt.close()
        
        # Imprimir tabela de importância
        print(f"\nImportância das Features - {modelo_nome}:")
        print(df_importancia.to_string(index=False))
        
        return df_importancia
    else:
        print(f"\nO modelo {modelo_nome} não suporta análise de importância de features.")
        return None

def analisar_erro_por_faixa(y_true, y_pred, modelo_nome, n_faixas=5):
    """Analisa erro do modelo em diferentes faixas de valores"""
    # Criar faixas de valores
    faixas = pd.qcut(y_true, n_faixas, labels=[f'Faixa {i+1}' for i in range(n_faixas)])
    
    # Calcular métricas por faixa
    df_erro = pd.DataFrame({
        'Valor Real': y_true,
        'Valor Predito': y_pred,
        'Faixa': faixas
    })
    
    # Calcular métricas por faixa
    metricas_por_faixa = df_erro.groupby('Faixa').agg({
        'Valor Real': ['count', 'mean', 'std'],
        'Valor Predito': ['mean', 'std'],
    }).round(2)
    
    # Calcular erro absoluto médio por faixa
    df_erro['Erro Absoluto'] = np.abs(df_erro['Valor Real'] - df_erro['Valor Predito'])
    mae_por_faixa = df_erro.groupby('Faixa')['Erro Absoluto'].mean().round(2)
    
    # Plotar gráfico de erro por faixa
    plt.figure(figsize=(12, 6))
    mae_por_faixa.plot(kind='bar')
    plt.title(f'{modelo_nome}: Erro Absoluto Médio por Faixa de Valor')
    plt.xlabel('Faixa de Valor')
    plt.ylabel('Erro Absoluto Médio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'result/erro_por_faixa_{modelo_nome.lower()}.png')
    plt.close()
    
    # Imprimir métricas por faixa
    print(f"\nAnálise de Erro por Faixa - {modelo_nome}:")
    print("\nMétricas por Faixa:")
    print(metricas_por_faixa)
    print("\nErro Absoluto Médio por Faixa:")
    print(mae_por_faixa)
    
    return metricas_por_faixa, mae_por_faixa

def avaliar_modelo(y_true, y_pred, modelo, X, modelo_nome):
    """
    Avalia modelo usando:
    1. Métricas de regressão
    2. Análise de resíduos
    3. Importância das features
    4. Erro por faixa
    """
    # Métricas básicas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nResultados do Modelo {modelo_nome}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Análise de resíduos
    residuos = analisar_residuos(y_true, y_pred, modelo_nome)
    
    # Análise de importância das features
    df_importancia = analisar_importancia_features(modelo, X, modelo_nome)
    
    # Análise de erro por faixa
    metricas_por_faixa, mae_por_faixa = analisar_erro_por_faixa(y_true, y_pred, modelo_nome)
    
    # Criar DataFrame com resultados
    resultados = {
        'Modelo': modelo_nome,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Média Resíduos': np.mean(residuos),
        'Std Resíduos': np.std(residuos)
    }
    
    return resultados

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
    """Cria visualizações das distribuições das variáveis"""
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
    plt.savefig('result/distribuicoes_variaveis.png')
    plt.close()

def visualizar_boxplots(df):
    """Cria boxplots para detectar outliers nas variáveis"""
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
    plt.savefig('result/boxplots_variaveis.png')
    plt.close()

def visualizar_correlacoes(df):
    """Cria matriz de correlação com heatmap"""
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('result/matriz_correlacao.png')
    plt.close()

def visualizar_predicoes(y_true, y_pred_rna, y_pred_rf):
    """Cria gráficos comparando valores reais e preditos dos modelos"""
    # Gráfico de dispersão para RNA
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred_rna, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('RNA: Valores Reais vs Preditos')
    plt.tight_layout()
    plt.savefig('result/rna_predicoes.png')
    plt.close()
    
    # Gráfico de dispersão para RF
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred_rf, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('RF: Valores Reais vs Preditos')
    plt.tight_layout()
    plt.savefig('result/rf_predicoes.png')
    plt.close()

def exportar_resultados_json(resultados_rna, resultados_rf, scores_rna, scores_rf, X_selected, y_test, y_pred_rna, y_pred_rf):
    """Exporta resultados dos modelos em formato JSON"""
    # Converter arrays numpy para listas para serialização JSON
    resultados_json = {
        "modelos": {
            "RNA": {
                "metricas": {
                    "MAE": float(resultados_rna["MAE"]),
                    "RMSE": float(resultados_rna["RMSE"]),
                    "R2": float(resultados_rna["R²"]),
                    "Media_Residuos": float(resultados_rna["Média Resíduos"]),
                    "Std_Residuos": float(resultados_rna["Std Resíduos"])
                },
                "validacao_cruzada": {
                    "RMSE_scores": scores_rna.tolist(),
                    "RMSE_medio": float(scores_rna.mean()),
                    "RMSE_std": float(scores_rna.std())
                }
            },
            "RF": {
                "metricas": {
                    "MAE": float(resultados_rf["MAE"]),
                    "RMSE": float(resultados_rf["RMSE"]),
                    "R2": float(resultados_rf["R²"]),
                    "Media_Residuos": float(resultados_rf["Média Resíduos"]),
                    "Std_Residuos": float(resultados_rf["Std Resíduos"])
                },
                "validacao_cruzada": {
                    "RMSE_scores": scores_rf.tolist(),
                    "RMSE_medio": float(scores_rf.mean()),
                    "RMSE_std": float(scores_rf.std())
                }
            }
        },
        "features_selecionadas": X_selected.columns.tolist(),
        "predicoes": {
            "valores_reais": y_test.tolist(),
            "predicoes_RNA": y_pred_rna.tolist(),
            "predicoes_RF": y_pred_rf.tolist()
        }
    }
    
    # Salvar resultados em arquivo JSON
    with open('result/resultados_modelos.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_json, f, ensure_ascii=False, indent=4)
    
    print("Resultados exportados com sucesso para 'result/resultados_modelos.json'")

if __name__ == "__main__":
    """
    Fluxo principal do script:
    1. Carrega e pré-processa dados
    2. Treina e otimiza modelos (RNA e Random Forest)
    3. Avalia e compara modelos
    4. Gera visualizações e relatórios
    """
    # Criar pasta para resultados
    criar_pasta_resultados()
    
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
        resultados_rna = avaliar_modelo(y_test, y_pred_rna, modelo_rna_otimizado, X_test, "RNA")
        
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
        resultados_rf = avaliar_modelo(y_test, y_pred_rf, modelo_rf_otimizado, X_test, "RF")
        
        # Visualizar predições
        print("\nCriando visualizações das predições...")
        visualizar_predicoes(y_test, y_pred_rna, y_pred_rf)
        
        # Atualizar relatório com resultados
        num_outliers = (X.shape[0] - X_clean.shape[0])
        porcentagem_outliers = (num_outliers / X.shape[0]) * 100
        atualizar_relatorio(num_outliers, porcentagem_outliers, X_clean.shape)
        
        # Criar DataFrame comparativo dos resultados
        df_resultados = pd.DataFrame([resultados_rna, resultados_rf])
        print("\nComparação dos Modelos:")
        print(df_resultados.to_string(index=False))
        
        # Salvar resultados em CSV
        df_resultados.to_csv('result/resultados_modelos.csv', index=False)
        
        # Exportar resultados em JSON
        exportar_resultados_json(resultados_rna, resultados_rf, scores_rna, scores_rf, X_selected, y_test, y_pred_rna, y_pred_rf)
        
        print("\nPré-processamento, seleção de características e treinamento dos modelos concluídos com sucesso!")
        print("Relatório atualizado com os resultados.")
        print("Todas as visualizações foram salvas na pasta 'result'.")
