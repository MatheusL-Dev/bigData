import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv(f'../base_de_dados/iris.data')

# Pré-processamento dos dados
data = data.dropna()  # Remover linhas com valores ausentes

# Separar as features (características) e o target (alvo)
X = data.drop('class', axis=1)
y = data['class']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)

# Criar o modelo Naive Bayes
model = GaussianNB()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)

print('Acurácia: {:.2%}'.format(accuracy))
