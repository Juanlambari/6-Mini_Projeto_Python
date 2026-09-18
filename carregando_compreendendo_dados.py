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