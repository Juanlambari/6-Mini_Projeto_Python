# Manipulação de dados e visualização
import re 
import pandas as pd
import numpy as np
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt

# Pré-Processamento e Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

nome_arquivo = 'dataset.csv'
df_dsa = pd.read_csv(nome_arquivo, index_col='review_id')

print("\nShape: ", df_dsa.shape)
print("\nPrimeiras linhas:\n",df_dsa.head())

print("\nlinhas aleátorias:\n",df_dsa.sample(5))

# Análise Exploratória
# Info
print("\nInformações:\n")
print(df_dsa.info())

print("\nVerificando valores ausentes:\n")
print(df_dsa.isnull().sum())

sns.countplot(x= 'sentimento', data=df_dsa)
plt.title('Distribuição das Classes de Sentimentos')
plt.show() # Dados estão balanceados

# Limpeza de Dados
# Remover linhas com valores ausentes
print(f"\nTamanho original do DataFrame: {len(df_dsa)}")
df_dsa.dropna(subset=['texto_review'], inplace = True)
print(f"Tamanho do DataFrame após remover nulos: {len(df_dsa)}") # .dropna() só funciona em casos específicos (como nesse que tem poucos registros e valores nulos não fariam sentido na aplicação), dropna em um contexto com muitos registros não é o melhor a se fazer, então deve-se procurar outras alternativas.

print(df_dsa.head())

# Função de limpeza de texto
def dsa_limpa_texto(texto):

    """
    Função completa de limpeza de texto:
    1. Converte para minúsculas.
    2. Remove acentos e cedilha.
    3. Remove pontuações, números e caracteres especiais.
    4. Remove espaços extras.
    """

    # Garante que o texto não seja nulo (caso haja algum NaN no DataFrame)
    if not isinstance(texto, str):
        return ""

    # --- PASSO 1: Normalizar e remover acentos ---
    # Normaliza para a forma 'NFKD' que separa o caractere da acentuação
    # e depois remove os acentos ()
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')

    # --- PASSO 2: Limpeza com Regex ---
    # Converter para minúsculas
    texto_limpo = texto_sem_acentos.lower()

    # Manter apenas letras e espaços. A remoção de acentos já foi feita.
    texto_limpo = re.sub(r'[^a-z\s]', '', texto_limpo)

    # Remover espaços extras
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()

    return texto_limpo

# aplica a função de limpeza
df_dsa['texto_limpo'] = df_dsa['texto_review'].apply(dsa_limpa_texto)

print(df_dsa.head())

# Engenharia de Atributos
# Mapear o sentimento para valores númericos (Engenharia de Atributos)
df_dsa['sentimento_label'] = df_dsa['sentimento'].map({'positivo': 1, 'negativo': 0})
print("\nDataFrame após a limpeza e mapeamento:\n")
df_dsa[['texto_limpo', 'sentimento_label']].head()

# Divisão em Dados de Treino de Teste

# Definir variáveis X(entrada) y(saída)
X = df_dsa['texto_limpo']
y = df_dsa['sentimento_label']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y )

# Pipeline de Modelagem Preditiva
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),

    ('scaler', StandardScaler(with_mean = False)),

    ('logreg', LogisticRegression(solver = 'liblinear', random_state = 42, max_iter = 1000))
])

# Definir o grid de hiperparâmetros para otimização
parametros_grid = {
    'tfidf__max_features': [500, 1000, 2000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'logreg__C': [0.1, 1, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__max_iter': [5000, 6000]
}
# Configurar o GridSearchCV
grid_search = GridSearchCV(
    pipeline,             # Pipeline com as etapas de pré-processamento e modelo
    parametros_grid,      # Dicionário com as combiinações de hiperparâmetros a serem testados
    cv = 5,               # Número de divisões para validação cruzada (5-fold cross validation)
    n_jobs = -1,          # Usa todos os núcleos disponíveis de processador para acelerar o processo
    scoring = 'accuracy', # Métrica usada para avaliar o desempenho de cada combinação (aqui, acurácia)
    verbose = 1           # Nível de detalhamento do output durante uma execução (1 exibe progresso básico)
)

# Treinando o Modelo
print("\nIniciando o treinamento do modelo com otimização de hiperparâmetros...\n")
grid_search.fit(X_treino, y_treino)

print("\nMelhores hiperparâmetros encontrados:\n")
print(grid_search.best_params_)

# Obter o melhor modelo
melhor_modelo_dsa = grid_search.best_estimator_
print(type(melhor_modelo_dsa))

# Previsões no conjunto de teste
y_pred = melhor_modelo_dsa.predict(X_teste)

# Calcular as métricas de avaliação
acuracia = accuracy_score(y_teste, y_pred)
report = classification_report(y_teste, y_pred, target_names= ['Negativo', 'Positivo'])

print(f"\nAcurácia do Modelo: {acuracia:.2%}\n")
print("Relatório de Classificação:\n")
print(report)

# Visualizar a Matriz de Confusao
cm = confusion_matrix(y_teste, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels = ['Negativo', 'Positivo'],
            yticklabels = ['Negativo', 'Positivo']
            )
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# Se estivermos satisfeitos com a perfomance do modelo, salvamos em disco
joblib.dump(melhor_modelo_dsa, 'modelo_sentimento_dsa_v1.joblib')
del melhor_modelo_dsa

# Carregar o modelo a partir do disco
modelo_dsa_deploy = joblib.load('modelo_sentimento_dsa_v1.joblib')
print(type(modelo_dsa_deploy))

# Criar novos dados para simular o uso em produção
novos_reviews = [
    "A bateria do celular não dura nada, péssima compra.",
    "Chegou antes do prazo e o produto é de ótima qualidade! Estou muito feliz.",
    "O serviço de atendimento foi rápido e eficiente.",
    "Não recomendo, veio faltando peças e a cor estava errada."
]

def dsa_prever_Sentimento(reviews):

    """
    Recebe uma lista de textos de review e retorna a previsão do sentimento.
    O objeto 'melhor_modelo_dsa' (pipeline) cuida de todos os passos internos.
    """

    # 1. 'reviews' entra no pipeline
    # 2. TF-IDF é aplicado internamente
    # 3. StandardScaler é aplicado internamente
    # 4. LogisticRegression faz a previsão
    previsoes = modelo_dsa_deploy.predict(reviews)

    # Mapeia o resultado numérico de volta para texto
    sentimentos = ['Negativo' if p == 0 else 'Positivo' for p in previsoes]

    # Exibe os resultados
    for review, sentimento in zip(reviews, sentimentos):
        print(f"\nReview: '{review}'\nSentimento Previsto: {sentimento}\n---")

# Executar a função de deploy com os novos dados
print("\n--- Iniciando Classificação de Novos Reviews (Deploy com Pipeline Completo) ---\n")
dsa_prever_Sentimento(novos_reviews)