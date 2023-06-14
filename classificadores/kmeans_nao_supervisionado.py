import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('../base_de_dados/abalone.data')

# Converter a coluna 'Sex' para valores numéricos
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Selecionar as colunas relevantes para a classificação
features = ['Sex', 'Length', 'Diameter', 'Height', 'Shell']

# Criar o modelo K-means
kmeans = KMeans(n_clusters=3, n_init=10) # Defina o número de clusters desejado

# Treinar o modelo
kmeans.fit(data[features])

# Obter as classificações dos clusters
labels = kmeans.labels_

# Adicionar as classificações ao DataFrame original
data['Cluster'] = labels

print(data)
