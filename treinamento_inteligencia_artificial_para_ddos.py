################################
#By: Rodrigo Schwartzhaupt Nunes
################################
#LOG
#10/09/2023 13:26
# Está Funcionando, AI treinada com 79 columas e dados novos com as mesmas 79 columas e com o mesmo filtro de informações que foi utilizado no treinamento 
# Retirei o ip de origem e de destino e tbm filtrei melhor a questão do vazios e dos infinitos, deixando sem falsos positivos
#------------------------------------------------------------------------------------------------------------------------------------------------------------


import time
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

####### ARGs ####### 
funcao = "treinamento"
####################

if funcao == "treinamento":
    ########## TRATANDO DADOS ##########

    # Importando arquivo
    data = pd.read_csv('Syn.csv', low_memory=False)

    # Mapear valores da última coluna para 0 se for "BENIGN", caso contrário, 1 e transformar ela em float
    data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: 0  if x == 'BENIGN' else 1).astype('float64')
    data[data.columns[-1]] = data[data.columns[-1]].astype(float)

    # Retira colunas da tabela
    remove_colunas = [0, 1, 2, 4, 7, 62, 85, 86]
    data_filtrada = data.drop(columns=data.columns[remove_colunas])

    #print (data_filtrada)

    # Ver nomes das colunas
    #nomes_das_colunas = data.columns.tolist()
    #print(nomes_das_colunas)
    #exit()

    # Convertendo valores da tabela para float, usa funcao
    #data_filtrada = data_filtrada.applymap(force_convert)

    #print (data_filtrada)

    # Calcula a matriz de correlação
    data_filtrada = data_filtrada.select_dtypes(include=['float64', 'int64'])
    data_filtrada = pd.get_dummies(data_filtrada)

    #Retirar linhas dos valores faltantes
    data_filtrada = data_filtrada.dropna()

    # Substitua os valores infinitos
    data_filtrada = data_filtrada.replace([np.inf, -np.inf], 3.4028235e+38)

    # Separando coluna y para prever
    y = data_filtrada.iloc[:, -1]

    # Retira a ultima coluna da tabela
    remove_colunas = [-1]
    data_filtrada = data_filtrada.drop(columns=data.columns[remove_colunas])

    # Escalonamento dos atributos
    scaler_data_filtrada = StandardScaler()
    data_filtrada = scaler_data_filtrada.fit_transform(data_filtrada)


    # Definindo valor de X
    x = data_filtrada

    #print(x)
    #exit()

    # Verificar valores faltantes
    #print(data_filtrada.isnull().sum())


    ########## INICIO COD AI ##########

    # Divisão em treinamento e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2)

    #GRAFICOS
    #numero_colunas = 4
    #grafico = px.scatter_matrix(data_filtrada, dimensions=data_filtrada.columns[:numero_colunas])
    #grafico.show()

    # Criação e treinamento do modelo
    modelo_floresta_aleatoria = RandomForestRegressor()
    modelo_floresta_aleatoria.fit(x_treino, y_treino)
    modelo_arvore_decisao = DecisionTreeRegressor()
    modelo_arvore_decisao.fit(x_treino, y_treino)

    # Realiza as previsões
    previsao_floresta_aleatoria = modelo_floresta_aleatoria.predict(x_teste)
    previsao_arvore_decisao = modelo_arvore_decisao.predict(x_teste)

    ########## INICIO GRAFICOS ##########

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Calculo percentual de acerto de previsao com o resultado de teste
    acuracia_floresta_aleatoria = r2_score(y_teste, previsao_floresta_aleatoria)
    acuracia_arvore_decisao = r2_score(y_teste, previsao_arvore_decisao)

    # Exibir a acurácia
    print(f"Acurácia Floresta Aleatória: {acuracia_floresta_aleatoria}")
    print(f"Acurácia Árvore de Decisão: {acuracia_arvore_decisao}")

    # Obtendo o nome da coluna de destino (ultima coluna)
    coluna_de_previsao = -1
    coluna_alvo = data.columns[coluna_de_previsao]

    # Avaliação do modelo em gráfico baseado y_teste
    tabela_aux = pd.DataFrame()
    tabela_aux['y_teste'] = y_teste
    tabela_aux['previsao_floresta_aleatoria'] = previsao_floresta_aleatoria
    tabela_aux['previsao_arvore_decisao'] = previsao_arvore_decisao
    tabela_aux[coluna_alvo] = y_teste.values  # Adicionando os valores da coluna de destino

    # Configuração dos subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig.suptitle(f'Comparação entre Valores Reais e Previsões - {coluna_alvo}')

    # Primeiro subplot para valores reais
    sns.lineplot(data=tabela_aux, x=tabela_aux.index, y='y_teste', ax=axes[0], label='Valores Reais', color='blue')
    axes[0].set_ylabel(coluna_alvo)

    # Segundo subplot para previsões da Floresta Aleatória
    sns.lineplot(data=tabela_aux, x=tabela_aux.index, y='previsao_floresta_aleatoria', ax=axes[1], label='Previsão Floresta Aleatória', color='orange')
    axes[1].set_ylabel(coluna_alvo)

    # Terceiro subplot para previsões da Árvore de Decisão
    sns.lineplot(data=tabela_aux, x=tabela_aux.index, y='previsao_arvore_decisao', ax=axes[2], label='Previsão Árvore de Decisão', color='green')
    axes[2].set_ylabel(coluna_alvo)

    plt.tight_layout()
    plt.show()

    
    # Exportar os modelos treinados para arquivos
    joblib.dump(modelo_floresta_aleatoria, 'modelo_floresta_aleatoria.pkl')
    joblib.dump(modelo_arvore_decisao, 'modelo_arvore_decisao.pkl')
