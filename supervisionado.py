# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns

# Carregando o dataset
df_new = pd.read_csv('archive/disaster-events new.csv')

# Filtrando para focar apenas na coluna "All disasters"
df_all_disasters = df_new[df_new['Entity'] == 'All disasters'].copy()

# Transformando a coluna "Disasters" em valores numéricos
df_all_disasters['target'] = (df_all_disasters['Disasters'] > 0).astype(int)

# Verificando as características disponíveis
print(df_all_disasters.columns)

# Separando as características e o alvo
X_new = df_all_disasters[['Year']]  # Usando apenas a coluna "Year" como característica
y_new = df_all_disasters['target']

# Dividindo os dados em conjuntos de treinamento e teste
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Criando um novo modelo
model_new = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=1000)

# Treinando o novo modelo
history_new = model_new.fit(X_train_new, y_train_new)

# Fazendo previsões no conjunto de teste com o novo modelo
y_pred_new = model_new.predict(X_test_new)

# Calculando as métricas do novo modelo
precision_new = precision_score(y_test_new, y_pred_new, average='macro')
recall_new = recall_score(y_test_new, y_pred_new, average='macro')
accuracy_new = accuracy_score(y_test_new, y_pred_new)

# Imprimindo as métricas do novo modelo
print(f'Novo Modelo - Precision: {precision_new}')
print(f'Novo Modelo - Recall: {recall_new}')
print(f'Novo Modelo - Accuracy: {accuracy_new}')

# Gerando a matriz de confusão do novo modelo
cm_new = confusion_matrix(y_test_new, y_pred_new)
print(f'Novo Modelo - Confusion Matrix: \n{cm_new}')

# Plotando a matriz de confusão do novo modelo
plt.figure(figsize=(8, 6))
sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues')
plt.title('Novo Modelo - Confusion Matrix')
plt.ylabel('Categoria Real')
plt.xlabel('Categoria Prevista')
plt.show()
