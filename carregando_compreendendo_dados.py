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