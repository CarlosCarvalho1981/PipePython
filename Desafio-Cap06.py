# Modelo preditivo de classificaçãp para prever o valor de uma variável binária (true ou false) a partir de dados numéricos
# Exercício desenvolvido no curso Formação Cientista de Dados (https://www.datascienceacademy.com.br/pages/formacao-cientista-de-dados)
# LinkedIn: https://www.linkedin.com/in/carlos-carvalho-93204b13/

# Import dos módulos
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# Dados de treino
n_train = 10
np.random.seed(0)
df_treino = pd.DataFrame({"var1": np.random.random(n_train), \
                          "var2": np.random.random(n_train), \
                          "var3": np.random.random(n_train), \
                          "var4": np.random.randint(0,2,n_train).astype(bool),\
                          "target": np.random.randint(0,2,n_train).astype(bool)})

# Dados de teste
n_test = 3
np.random.seed(1)
df_teste = pd.DataFrame({"var1": np.random.random(n_test), \
                         "var2": np.random.random(n_test), \
                         "var3": np.random.random(n_test), \
                         "var4": np.random.randint(0,2,n_test).astype(bool),\
                         "target": np.random.randint(0,2,n_test).astype(bool)})

# Redução da dimensionalidade para 3 componentes
pca = PCA(n_components = 3)


# Aplicação do PCA aos datasets
pcaDF_treino = pca.fit_transform(df_treino.drop("target", axis = 1))
pcaDF_teste = pca.transform(df_teste.drop("target", axis = 1))

# Dataframes com os resultados da redução de dimensionalidade
ComponentesTreino = pd.DataFrame(pcaDF_treino)
ComponentesTeste = pd.DataFrame(pcaDF_teste)

# Modelo de regressão logística
modelo = LogisticRegression()

# Criação do pipeline
pipe = Pipeline([('pca', pca), ('RegLog', modelo)])

# Treinamento encadeando o pca com a regressão logística. Os argumentos do método fit são os dados de entrada e saída.
pipe.fit(ComponentesTreino, df_treino["target"])

# Previsões com o modelo treinado
previsoes = pipe.predict(ComponentesTeste)

# Impr"imindo as previsões
print("Previsões: \n")
print(previsoes)

