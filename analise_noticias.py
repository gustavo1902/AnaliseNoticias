import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('NEWS_API_KEY') 

# Define a consulta para buscar notícias
QUERY = 'inteligência artificial'

def fetch_news(api_key, query):
    """
    Busca notícias na NewsAPI.
    """
    url = f'https://newsapi.org/v2/everything?q={query}&language=pt&sortBy=publishedAt&apiKey={api_key}'
    
    print(f"Buscando notícias sobre '{query}'...")
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        
        if data['status'] == 'ok' and data['totalResults'] > 0:
            print(f"Encontrados {len(data['articles'])} artigos.")
            return data['articles']
        elif data['totalResults'] == 0:
            print("Nenhum artigo encontrado para esta busca.")
            return []
        else:
            print(f"Erro da API: {data.get('message', 'Erro desconhecido')}")
            return []

    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP: {http_err}")
        print(f"Corpo da resposta: {response.text}")
        return []
    except Exception as err:
        print(f"Ocorreu um erro: {err}")
        return []

def analyze_sentiment(text):
    """
    Analisa o sentimento de um texto usando VADER e retorna o rótulo e o score.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        return 'Positivo', compound_score
    elif compound_score <= -0.05:
        return 'Negativo', compound_score
    else:
        return 'Neutro', compound_score

def plot_results(df):
    """
    Cria e exibe um gráfico com os resultados da análise de sentimento.
    """
    sns.set_theme(style="whitegrid")
    sentiment_counts = df['Sentimento'].value_counts()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['#4CAF50', '#F44336', '#FFC107'])
    
    plt.title(f'Análise de Sentimento para: "{QUERY.title()}"', fontsize=16)
    plt.xlabel('Sentimento', fontsize=12)
    plt.ylabel('Número de Notícias', fontsize=12)
    plt.xticks(fontsize=11)
    
    output_filename = 'grafico_sentimentos.png'
    plt.savefig(output_filename)
    print(f"\nGráfico salvo como '{output_filename}' no diretório: {os.getcwd()}")
    
    plt.show()

if __name__ == "__main__":
    if not API_KEY:
        print("ERRO: Chave da API não encontrada.")
        print("Verifique se você criou o arquivo .env e se ele contém a variável NEWS_API_KEY.")
    else:
        articles = fetch_news(API_KEY, QUERY)
        
        if articles:
            results = []
            for article in articles:
                title = article.get('title', '')
                if title:
                    sentiment_label, sentiment_score = analyze_sentiment(title)
                    results.append({
                        'Título': title,
                        'Fonte': article['source']['name'],
                        'Sentimento': sentiment_label,
                        'Score (Compound)': f"{sentiment_score:.2f}",
                        'URL': article['url']
                    })
            
            df = pd.DataFrame(results)
            
            # Exibir os resultados
            print("\n--- Análise de Sentimento das Notícias ---")
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_colwidth', 80)
            print(df[['Título', 'Fonte', 'Sentimento', 'Score (Compound)']])
    
            plot_results(df)