# Classificador de Spam com Keras e PLN

Este projeto implementa um classificador de spam utilizando uma rede neural simples construída com Keras. O objetivo é categorizar mensagens de texto (simulando e-mails ou SMS) como "ham" (legítimas) ou "spam", aplicando técnicas fundamentais de Processamento de Linguagem Natural (PLN).

## Visão Geral

A detecção de spam é um problema clássico de classificação de texto em PLN. Este projeto aborda o problema da seguinte forma:
1.  **Carregamento e Pré-processamento de Dados:** Um dataset de mensagens rotuladas como "ham" ou "spam" é carregado. É realizado um balanceamento das classes para garantir um treinamento mais eficaz, já que os datasets de spam geralmente são desbalanceados.
2.  **Codificação de Rótulos:** As categorias "ham" e "spam" são convertidas em representações numéricas (0 e 1).
3.  **Divisão em Conjuntos de Treinamento e Teste:** Os dados são divididos para treinamento e avaliação do modelo.
4.  **Tokenização e Sequenciamento:** As mensagens de texto são convertidas em sequências numéricas usando um `Tokenizer` do Keras, que mapeia palavras para índices. As sequências são então preenchidas (padded) para terem o mesmo comprimento.
5.  **Construção do Modelo de Rede Neural:** Uma rede neural simples é criada com:
    * Uma `Embedding` layer para criar representações densas das palavras.
    * Uma `Flatten` layer para achatar a saída da camada de embedding.
    * Camadas `Dense` (densas) com ativação ReLU e Sigmoid para a classificação binária.
    * Uma camada `Dropout` para regularização.
6.  **Treinamento e Avaliação:** O modelo é compilado com `mean_squared_error` como função de perda e `adam` como otimizador, e então treinado e avaliado com base na acurácia e na matriz de confusão.

## Estrutura do Projeto

* `NLPRna_.ipynb`: O notebook Jupyter que contém todo o código para as etapas de pré-processamento, construção do modelo, treinamento e avaliação.
* `spam.csv`: O dataset de exemplo contendo mensagens rotuladas como "ham" ou "spam". (Este arquivo deve ser fornecido ou ter uma instrução para download).

## Tecnologias Utilizadas

* **Python 3**
* **Pandas**: Para manipulação e análise de dados (DataFrames).
* **NumPy**: Para operações numéricas.
* **Scikit-learn**: Para divisão de dados (train_test_split), codificação de rótulos (LabelEncoder) e avaliação (confusion_matrix).
* **TensorFlow/Keras**: Para construção e treinamento da rede neural.

## Resultados

O notebook apresentará:
* As contagens iniciais das categorias "ham" e "spam" e a distribuição após o balanceamento.
* A perda (loss) e a acurácia (accuracy) do modelo nos conjuntos de treinamento e teste.
* Exemplos de previsões do modelo (probabilidades e classificação binária).
* Uma matriz de confusão que detalha os verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
