import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Configurações de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def criar_pasta_resultado():
    """
    Cria a pasta 'grafico-comparacao' se ela não existir
    """
    if not os.path.exists('grafico-comparacao'):
        os.makedirs('grafico-comparacao')
        print("Pasta 'grafico-comparacao' criada com sucesso!")

def carregar_resultados(caminho_arquivo='result/resultados_modelos.json'):
    """
    Carrega os resultados dos modelos
    """
    try:
        with open(caminho_arquivo, 'r') as file:
            resultados = json.load(file)
        return resultados
    except Exception as e:
        print(f"Erro ao carregar os resultados: {e}")
        # Tentar carregar do CSV como alternativa
        try:
            resultados_df = pd.read_csv('result/resultados_modelos.csv')
            print("Resultados carregados do CSV com sucesso!")
            return resultados_df
        except Exception as e2:
            print(f"Erro ao carregar os resultados do CSV: {e2}")
            return None

def comparar_metricas(resultados):
    """
    Gera um gráfico comparativo das métricas dos modelos
    """
    if isinstance(resultados, pd.DataFrame):
        # Se os resultados foram carregados do CSV
        df_metricas = resultados.set_index('Modelo')
    else:
        # Se os resultados foram carregados do JSON
        metricas = {
            'RNA': {
                'MAE': resultados['modelos']['RNA']['metricas']['MAE'],
                'RMSE': resultados['modelos']['RNA']['metricas']['RMSE'],
                'R²': resultados['modelos']['RNA']['metricas']['R2']
            },
            'RF': {
                'MAE': resultados['modelos']['RF']['metricas']['MAE'],
                'RMSE': resultados['modelos']['RF']['metricas']['RMSE'],
                'R²': resultados['modelos']['RF']['metricas']['R2']
            }
        }
        df_metricas = pd.DataFrame(metricas).T

    # Criando o gráfico de barras para cada métrica
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MAE (menor é melhor)
    sns.barplot(x=df_metricas.index, y='MAE', data=df_metricas, ax=axes[0], palette='viridis')
    axes[0].set_title('Erro Médio Absoluto (MAE)')
    axes[0].set_ylabel('Valor (menor é melhor)')
    
    # RMSE (menor é melhor)
    sns.barplot(x=df_metricas.index, y='RMSE', data=df_metricas, ax=axes[1], palette='viridis')
    axes[1].set_title('Raiz do Erro Quadrático Médio (RMSE)')
    axes[1].set_ylabel('Valor (menor é melhor)')
    
    # R² (maior é melhor)
    sns.barplot(x=df_metricas.index, y='R²', data=df_metricas, ax=axes[2], palette='viridis')
    axes[2].set_title('Coeficiente de Determinação (R²)')
    axes[2].set_ylabel('Valor (maior é melhor)')
    
    plt.tight_layout()
    plt.savefig('grafico-comparacao/comparacao_metricas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráfico de comparação de métricas gerado com sucesso!")

def visualizar_predicoes_comparadas(resultados):
    """
    Gera um gráfico comparativo das predições dos modelos
    """
    if isinstance(resultados, pd.DataFrame):
        # Se não temos os dados de predição, não podemos gerar este gráfico
        print("Dados de predição não disponíveis no CSV. Pulando gráfico de predições comparadas.")
        return
    
    # Extraindo dados de predição
    y_test = np.array(resultados['predicoes']['valores_reais'])
    y_pred_rna = np.array(resultados['predicoes']['predicoes_RNA'])
    y_pred_rf = np.array(resultados['predicoes']['predicoes_RF'])
    
    # Criando DataFrame para visualização
    df_pred = pd.DataFrame({
        'Valor Real': y_test,
        'RNA': y_pred_rna,
        'Random Forest': y_pred_rf
    })
    
    # Ordenando por valor real para melhor visualização
    df_pred = df_pred.sort_values('Valor Real')
    df_pred = df_pred.reset_index(drop=True)
    
    # Plotando comparação de predições
    plt.figure(figsize=(14, 8))
    plt.plot(df_pred.index, df_pred['Valor Real'], 'o-', label='Valor Real', linewidth=2)
    plt.plot(df_pred.index, df_pred['RNA'], 'o-', label='RNA', alpha=0.7)
    plt.plot(df_pred.index, df_pred['Random Forest'], 'o-', label='Random Forest', alpha=0.7)
    
    plt.title('Comparação das Predições dos Modelos', fontsize=16)
    plt.xlabel('Índice da Amostra (ordenado por valor real)', fontsize=14)
    plt.ylabel('Preço da Casa (MEDV)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('grafico-comparacao/comparacao_predicoes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráfico de comparação de predições gerado com sucesso!")

def criar_grafico_validacao_cruzada(resultados):
    """
    Gera um gráfico comparativo dos resultados da validação cruzada
    """
    if isinstance(resultados, pd.DataFrame):
        print("Dados de validação cruzada não disponíveis no CSV. Pulando gráfico de validação cruzada.")
        return
    
    try:
        # Extraindo scores de validação cruzada
        rmse_cv_rna = resultados['modelos']['RNA']['validacao_cruzada']['RMSE_scores']
        rmse_cv_rf = resultados['modelos']['RF']['validacao_cruzada']['RMSE_scores']
        
        # Criando DataFrame para visualização
        df_cv = pd.DataFrame({
            'Fold': range(1, len(rmse_cv_rna) + 1),
            'RNA': rmse_cv_rna,
            'Random Forest': rmse_cv_rf
        })
        
        # Convertendo para formato longo para seaborn
        df_cv_long = pd.melt(df_cv, id_vars=['Fold'], 
                            value_vars=['RNA', 'Random Forest'],
                            var_name='Modelo', value_name='RMSE')
        
        # Plotando comparação de validação cruzada
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df_cv_long, x='Fold', y='RMSE', hue='Modelo', marker='o', linewidth=2)
        
        # Adicionando médias
        plt.axhline(y=resultados['modelos']['RNA']['validacao_cruzada']['RMSE_medio'], 
                   color='blue', linestyle='--', alpha=0.7,
                   label=f"RNA Média: {resultados['modelos']['RNA']['validacao_cruzada']['RMSE_medio']:.2f}")
        
        plt.axhline(y=resultados['modelos']['RF']['validacao_cruzada']['RMSE_medio'], 
                   color='orange', linestyle='--', alpha=0.7,
                   label=f"RF Média: {resultados['modelos']['RF']['validacao_cruzada']['RMSE_medio']:.2f}")
        
        plt.title('Comparação dos Resultados da Validação Cruzada (RMSE)', fontsize=16)
        plt.xlabel('Fold', fontsize=14)
        plt.ylabel('RMSE (menor é melhor)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig('grafico-comparacao/validacao_cruzada.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Gráfico de comparação de validação cruzada gerado com sucesso!")
    except Exception as e:
        print(f"Erro ao gerar gráfico de validação cruzada: {e}")

def criar_grafico_importancia_features(resultados):
    """
    Gera um gráfico com as features mais importantes do modelo Random Forest
    """
    if isinstance(resultados, pd.DataFrame):
        print("Dados de importância de features não disponíveis no CSV. Pulando gráfico de importância de features.")
        return
    
    try:
        # Verificar se temos os dados de importância
        if 'importancia_features' not in resultados:
            print("Dados de importância de features não encontrados no JSON.")
            return
        
        # Extraindo importância das features
        features = list(resultados['importancia_features'].keys())
        importancias = list(resultados['importancia_features'].values())
        
        # Criando DataFrame para visualização
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importância': importancias
        })
        
        # Ordenando por importância
        df_imp = df_imp.sort_values('Importância', ascending=False)
        
        # Plotando importância das features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importância', y='Feature', data=df_imp, palette='viridis')
        plt.title('Importância das Features (Random Forest)', fontsize=16)
        plt.xlabel('Importância Relativa', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grafico-comparacao/importancia_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Gráfico de importância de features gerado com sucesso!")
    except Exception as e:
        print(f"Erro ao gerar gráfico de importância de features: {e}")

if __name__ == "__main__":
    # Criar pasta para resultados
    criar_pasta_resultado()
    
    # Carregar resultados
    resultados = carregar_resultados()
    
    if resultados is not None:
        # Gerar gráficos comparativos
        comparar_metricas(resultados)
        visualizar_predicoes_comparadas(resultados)
        criar_grafico_validacao_cruzada(resultados)
        criar_grafico_importancia_features(resultados)
        
        print("\nTodos os gráficos comparativos foram gerados com sucesso!")
    else:
        print("Não foi possível gerar os gráficos comparativos devido a erros na carga dos resultados.") 