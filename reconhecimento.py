import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# URLs dos arquivos raw no GitHub
url_X_train = "https://raw.githubusercontent.com/brendatrindade/Reconhecimento-de-atividade-humana/main/datasheet/UCI%20HAR/UCI%20HAR%20Dataset/train/X_train.txt"
url_X_test = "https://raw.githubusercontent.com/brendatrindade/Reconhecimento-de-atividade-humana/main/datasheet/UCI%20HAR/UCI%20HAR%20Dataset/test/X_test.txt"
url_y_train = "https://raw.githubusercontent.com/brendatrindade/Reconhecimento-de-atividade-humana/main/datasheet/UCI%20HAR/UCI%20HAR%20Dataset/train/y_train.txt"
url_y_test = "https://raw.githubusercontent.com/brendatrindade/Reconhecimento-de-atividade-humana/main/datasheet/UCI%20HAR/UCI%20HAR%20Dataset/test/y_test.txt"
url_subject_train = "https://raw.githubusercontent.com/brendatrindade/Reconhecimento-de-atividade-humana/main/datasheet/UCI%20HAR/UCI%20HAR%20Dataset/train/subject_train.txt"
url_subject_test = "https://raw.githubusercontent.com/brendatrindade/Reconhecimento-de-atividade-humana/main/datasheet/UCI%20HAR/UCI%20HAR%20Dataset/test/subject_test.txt"

# Carregar os dados diretamente do GitHub
X_train = pd.read_csv(url_X_train, sep='\s+', header=None)
X_test = pd.read_csv(url_X_test, sep='\s+', header=None)

y_train = pd.read_csv(url_y_train, sep='\s+', header=None)
y_test = pd.read_csv(url_y_test, sep='\s+', header=None)

subject_train = pd.read_csv(url_subject_train, sep='\s+', header=None)
subject_test = pd.read_csv(url_subject_test, sep='\s+', header=None)

# Combinar os dados de treino e teste
X_data = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_data = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
subjects = pd.concat([subject_train, subject_test], axis=0).reset_index(drop=True)

# Exibir informações básicas do dataset
print(f"Número de amostras: {X_data.shape[0]}")
print(f"Número de features: {X_data.shape[1]}")
print(f"Número de atividades: {y_data[0].nunique()}")
print(f"Número de sujeitos: {subjects[0].nunique()}")

# Normalizar os dados para a análise de PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# Aplicar PCA para reduzir a dimensionalidade para 2 componentes principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Criar DataFrame para visualização
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Activity'] = y_data[0].values

# Visualizar as duas principais componentes por atividade, a projeção PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Activity', palette='Set1', legend='full', alpha=0.7)
plt.title('Projeção PCA das Atividades')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Atividade')
plt.show()

# Testar diferentes valores de K para encontrar o número ideal de clusters
inertia = []
silhouette_scores = []
k_values = range(2, 10)  # Valores de K entre 2 e 9

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Visualizar o método do cotovelo e silhouette score
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Método do cotovelo
ax[0].plot(k_values, inertia, '-o', color='blue')
ax[0].set_title('Método do Cotovelo')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Inércia')

# Silhouette Score
ax[1].plot(k_values, silhouette_scores, '-o', color='green')
ax[1].set_title('Silhouette Score para Diferentes K')
ax[1].set_xlabel('Número de Clusters (K)')
ax[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Escolher o número de clusters ideal (K=6 baseado no número de atividades)
k_optimal = 6

# Aplicar o K-means com K=6
kmeans = KMeans(n_clusters=k_optimal, random_state=42, init='k-means++', n_init=10)
kmeans.fit(X_pca)

# Adicionar os rótulos de cluster ao DataFrame PCA
pca_df['Cluster'] = kmeans.labels_

# Visualizar os clusters gerados pelo K-means
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', legend='full', alpha=0.7)
plt.title('Clusters Gerados pelo K-means (K=6)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.show()

# Avaliar o desempenho dos clusters com silhouette score
silhouette_avg= silhouette_score(X_pca, kmeans.labels_)
print(f"Silhouette Score para K=6: {silhouette_avg}")
