import os
import json
import pandas as pd
import base64
import markdown
from datetime import datetime

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

def codificar_imagem_para_html(caminho_imagem):
    """
    Codifica uma imagem para base64 para incluir diretamente no HTML
    """
    try:
        with open(caminho_imagem, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"Erro ao codificar imagem {caminho_imagem}: {e}")
        return ""

def criar_tabela_metricas(resultados):
    """
    Cria uma tabela HTML com as métricas dos modelos
    """
    if isinstance(resultados, pd.DataFrame):
        # Se os resultados foram carregados do CSV
        df_metricas = resultados.copy()
    else:
        # Se os resultados foram carregados do JSON
        metricas = {
            'Modelo': ['RNA', 'Random Forest'],
            'MAE': [
                resultados['modelos']['RNA']['metricas']['MAE'],
                resultados['modelos']['RF']['metricas']['MAE']
            ],
            'RMSE': [
                resultados['modelos']['RNA']['metricas']['RMSE'],
                resultados['modelos']['RF']['metricas']['RMSE']
            ],
            'R²': [
                resultados['modelos']['RNA']['metricas']['R2'],
                resultados['modelos']['RF']['metricas']['R2']
            ]
        }
        df_metricas = pd.DataFrame(metricas)
    
    # Formatando os valores
    df_metricas['MAE'] = df_metricas['MAE'].round(2)
    df_metricas['RMSE'] = df_metricas['RMSE'].round(2)
    df_metricas['R²'] = df_metricas['R²'].round(2)
    
    # Convertendo para HTML
    tabela_html = df_metricas.to_html(index=False, classes='table table-striped')
    return tabela_html

def gerar_html_apresentacao(resultados):
    """
    Gera uma apresentação em HTML com os resultados do projeto
    """
    # Carregando imagens como base64
    img_comparacao_metricas = codificar_imagem_para_html('grafico-comparacao/comparacao_metricas.png')
    img_comparacao_predicoes = codificar_imagem_para_html('grafico-comparacao/comparacao_predicoes.png')
    img_validacao_cruzada = codificar_imagem_para_html('grafico-comparacao/validacao_cruzada.png')
    
    # Carregando imagens dos resultados
    img_matriz_correlacao = codificar_imagem_para_html('result/matriz_correlacao.png')
    img_distribuicoes = codificar_imagem_para_html('result/distribuicoes_variaveis.png')
    img_boxplots = codificar_imagem_para_html('result/boxplots_variaveis.png')
    img_residuos_rna = codificar_imagem_para_html('result/analise_residuos_rna.png')
    img_residuos_rf = codificar_imagem_para_html('result/analise_residuos_rf.png')
    
    # Criando tabela de métricas
    tabela_metricas = criar_tabela_metricas(resultados)
    
    # Lendo o conteúdo do relatório
    try:
        with open('relatorio.md', 'r', encoding='utf-8') as file:
            relatorio_md = file.read()
            relatorio_html = markdown.markdown(relatorio_md)
    except Exception as e:
        print(f"Erro ao ler o relatório: {e}")
        relatorio_html = "<p>Erro ao carregar o relatório.</p>"
    
    # Criando o HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Projeto Boston Housing - Aprendizado Supervisionado</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                color: #333;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 2rem 0;
                text-align: center;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }}
            .section {{
                margin-bottom: 3rem;
                padding: 1.5rem;
                border-radius: 8px;
                background-color: #f8f9fa;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.5rem;
                margin-bottom: 1.5rem;
            }}
            .img-fluid {{
                max-width: 100%;
                height: auto;
                margin: 1rem 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .table {{
                margin-top: 1rem;
            }}
            .footer {{
                background-color: #2c3e50;
                color: white;
                text-align: center;
                padding: 1rem 0;
                margin-top: 2rem;
            }}
            .nav-tabs {{
                margin-bottom: 1rem;
            }}
            .tab-content {{
                padding: 1rem;
                background-color: white;
                border: 1px solid #dee2e6;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>Projeto Boston Housing - Aprendizado Supervisionado</h1>
                <p>Comparação entre RNA e Random Forest para predição de preços de imóveis</p>
            </div>
        </div>
        
        <div class="container">
            <div class="section">
                <h2>1. Introdução</h2>
                <p>Este projeto implementa e compara diferentes modelos de aprendizado supervisionado para prever preços de casas usando o dataset Boston Housing. O objetivo é analisar o comportamento dos modelos, justificar as decisões tomadas e extrair insights valiosos sobre o problema.</p>
                <p>Os modelos implementados são:</p>
                <ul>
                    <li><strong>Rede Neural Artificial (RNA)</strong></li>
                    <li><strong>Random Forest (RF)</strong></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>2. Análise Exploratória dos Dados</h2>
                
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="correlacao-tab" data-bs-toggle="tab" data-bs-target="#correlacao" type="button" role="tab" aria-controls="correlacao" aria-selected="true">Matriz de Correlação</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="distribuicoes-tab" data-bs-toggle="tab" data-bs-target="#distribuicoes" type="button" role="tab" aria-controls="distribuicoes" aria-selected="false">Distribuições</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="boxplots-tab" data-bs-toggle="tab" data-bs-target="#boxplots" type="button" role="tab" aria-controls="boxplots" aria-selected="false">Boxplots</button>
                    </li>
                </ul>
                <div class="tab-content" id="myTabContent">
                    <div class="tab-pane fade show active" id="correlacao" role="tabpanel" aria-labelledby="correlacao-tab">
                        <img src="{img_matriz_correlacao}" class="img-fluid" alt="Matriz de Correlação">
                        <p>A matriz de correlação mostra as relações entre as variáveis. Observamos forte correlação negativa entre LSTAT e MEDV, e correlação positiva entre RM e MEDV.</p>
                    </div>
                    <div class="tab-pane fade" id="distribuicoes" role="tabpanel" aria-labelledby="distribuicoes-tab">
                        <img src="{img_distribuicoes}" class="img-fluid" alt="Distribuições das Variáveis">
                        <p>As distribuições das variáveis mostram que algumas, como CRIM e LSTAT, apresentam distribuição assimétrica, enquanto outras, como RM, mostram distribuição mais normal.</p>
                    </div>
                    <div class="tab-pane fade" id="boxplots" role="tabpanel" aria-labelledby="boxplots-tab">
                        <img src="{img_boxplots}" class="img-fluid" alt="Boxplots das Variáveis">
                        <p>Os boxplots ajudam a identificar outliers nas variáveis. Observamos que variáveis como CRIM e LSTAT possuem vários outliers.</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>3. Comparação dos Modelos</h2>
                <h3>3.1. Métricas de Performance</h3>
                <div class="row">
                    <div class="col-md-6">
                        {tabela_metricas}
                    </div>
                    <div class="col-md-6">
                        <img src="{img_comparacao_metricas}" class="img-fluid" alt="Comparação das Métricas">
                    </div>
                </div>
                
                <h3>3.2. Predições</h3>
                <img src="{img_comparacao_predicoes}" class="img-fluid" alt="Comparação das Predições">
                <p>O gráfico acima mostra as predições dos modelos comparadas com os valores reais. Observamos que o Random Forest tende a seguir mais de perto os valores reais, especialmente nos extremos.</p>
                
                <h3>3.3. Validação Cruzada</h3>
                <img src="{img_validacao_cruzada}" class="img-fluid" alt="Validação Cruzada">
                <p>A validação cruzada mostra a consistência dos modelos em diferentes folds. O Random Forest apresenta menor RMSE médio e menor variabilidade entre os folds.</p>
                
                <h3>3.4. Análise de Resíduos</h3>
                <div class="row">
                    <div class="col-md-6">
                        <h4>RNA</h4>
                        <img src="{img_residuos_rna}" class="img-fluid" alt="Análise de Resíduos RNA">
                    </div>
                    <div class="col-md-6">
                        <h4>Random Forest</h4>
                        <img src="{img_residuos_rf}" class="img-fluid" alt="Análise de Resíduos RF">
                    </div>
                </div>
                <p>A análise de resíduos mostra que o Random Forest apresenta distribuição mais simétrica dos resíduos e menor variabilidade.</p>
            </div>
            
            <div class="section">
                <h2>4. Conclusões</h2>
                <p>Com base nos resultados obtidos, podemos concluir que:</p>
                <ul>
                    <li>O Random Forest apresentou melhor performance em todas as métricas avaliadas (MAE, RMSE e R²).</li>
                    <li>O Random Forest mostrou maior estabilidade nas predições e resíduos mais bem comportados.</li>
                    <li>A RNA, apesar de performance inferior, ainda apresentou resultados aceitáveis.</li>
                    <li>O pré-processamento adequado dos dados teve impacto significativo nos resultados.</li>
                    <li>A interpretabilidade do Random Forest é uma vantagem adicional para aplicações práticas.</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>5. Relatório Completo</h2>
                <div class="accordion" id="accordionRelatorio">
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingRelatorio">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseRelatorio" aria-expanded="false" aria-controls="collapseRelatorio">
                                Clique para expandir o relatório completo
                            </button>
                        </h2>
                        <div id="collapseRelatorio" class="accordion-collapse collapse" aria-labelledby="headingRelatorio" data-bs-parent="#accordionRelatorio">
                            <div class="accordion-body">
                                {relatorio_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="container">
                <p>Projeto de Aprendizado Supervisionado - Boston Housing Dataset</p>
                <p>Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Salvando o HTML
    try:
        with open('apresentacao.html', 'w', encoding='utf-8') as file:
            file.write(html)
        print("Apresentação HTML gerada com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao gerar apresentação HTML: {e}")
        return False

if __name__ == "__main__":
    # Carregar resultados
    resultados = carregar_resultados()
    
    if resultados is not None:
        # Gerar apresentação HTML
        gerar_html_apresentacao(resultados)
    else:
        print("Não foi possível gerar a apresentação HTML devido a erros na carga dos resultados.") 