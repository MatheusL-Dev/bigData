import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('../base_de_dados/raisin_dataset.data')

# Pré-processamento dos dados
data = data.dropna()  # Remover linhas com valores ausentes

# Separar as features (características) e o target (alvo)
X = data.drop('class', axis=1)
y = data['class']

# Codificar as variáveis categóricas
categorical_columns = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_columns])
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
X = pd.concat([X.drop(categorical_columns, axis=1), X_encoded], axis=1)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=91)

# Criar o modelo Naive Bayes
model = GaussianNB()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)

print('Acurácia: {:.2%}'.format(accuracy))
