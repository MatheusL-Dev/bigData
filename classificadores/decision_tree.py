import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import graphviz
from sklearn.tree import export_graphviz


data = pd.read_csv('../base_de_dados/iris.data')

# Pré-processamento dos dados
data = data.dropna()  # Remover linhas com valores ausentes

# Separar os recursos (features) e os rótulos (labels)
X = data.drop('class', axis=1)
y = data['class']

# Codificar as classes em números
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Criar o modelo da árvore de decisão
model = DecisionTreeClassifier()

# Treinar o modelo
model.fit(X, y)

# Exportar a estrutura da árvore de decisão em formato DOT
dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)

# Gerar o gráfico da árvore de decisão
graph = graphviz.Source(dot_data)

# Exibir o gráfico no terminal
graph.format = 'png'  # Você também pode usar outros formatos, como 'pdf' ou 'svg'
graph.render(filename='decision_tree', view=True)
