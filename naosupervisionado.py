import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o conjunto de dados
df = pd.read_csv('archive/National_Olympic_Committee_2022_medals.csv')

# Substituir vírgulas em todas as colunas do DataFrame
df = df.replace(',', '', regex=True)

# Converter as colunas para float
df[['SOG_total_medals', 'WOG_total_medals']] = df[['SOG_total_medals', 'WOG_total_medals']].astype(float)

# Pré-processamento dos dados
# Normalizar os dados se necessário
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['SOG_total_medals', 'WOG_total_medals']])

# Implementar a Rede Neural Não Supervisionada - KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Avaliar o desempenho do modelo de agrupamento
silhouette_avg = silhouette_score(df_scaled, df['cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Visualizar os resultados
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('Total Medals in Summer Olympics (normalized)')
plt.ylabel('Total Medals in Winter Olympics (normalized)')
plt.title('Clusters of Countries by Medal Count')
plt.show()

# Implementar diferentes arquiteturas de rede
# Exemplo: Rede com uma camada oculta
kmeans_1 = KMeans(n_clusters=3, random_state=42)
df['cluster_1'] = kmeans_1.fit_predict(df_scaled)

# Exemplo: Rede com duas camadas ocultas
kmeans_2 = KMeans(n_clusters=3, random_state=42, n_init=20)
df['cluster_2'] = kmeans_2.fit_predict(df_scaled)

# Avaliar o desempenho das diferentes arquiteturas de rede
silhouette_avg_1 = silhouette_score(df_scaled, df['cluster_1'])
silhouette_avg_2 = silhouette_score(df_scaled, df['cluster_2'])
print(f'Silhouette Score for first architecture: {silhouette_avg_1:.2f}')
print(f'Silhouette Score for second architecture: {silhouette_avg_2:.2f}')

# Plotar a matriz de confusão para o cluster original
cm = pd.crosstab(df['SOG_total_medals'], df['cluster'], rownames=['SOG_total_medals'], colnames=['Cluster'])
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SOG_total_medals and Clusters')
plt.ylabel('Total Medals in Summer Olympics')
plt.xlabel('Cluster')
plt.show()

# Plotar a matriz de confusão para o cluster com uma camada oculta
cm_1 = pd.crosstab(df['SOG_total_medals'], df['cluster_1'], rownames=['SOG_total_medals'], colnames=['Cluster_1'])
plt.figure(figsize=(10, 6))
sns.heatmap(cm_1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SOG_total_medals and Cluster_1')
plt.ylabel('Total Medals in Summer Olympics')
plt.xlabel('Cluster_1')
plt.show()

# Plotar a matriz de confusão para o cluster com duas camadas ocultas
cm_2 = pd.crosstab(df['SOG_total_medals'], df['cluster_2'], rownames=['SOG_total_medals'], colnames=['Cluster_2'])
plt.figure(figsize=(10, 6))
sns.heatmap(cm_2, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SOG_total_medals and Cluster_2')
plt.ylabel('Total Medals in Summer Olympics')
plt.xlabel('Cluster_2')
plt.show()
