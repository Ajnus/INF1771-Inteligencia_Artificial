import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import time

Xtreino = pd.read_csv('treino.csv')
Xteste = pd.read_csv('teste.csv')

#### Inicio do Treino
trainingStartTime = time.time()
# Cria uma matriz com as palavras que aparecem no conjunto de treinos
# Insere os casos de treino com resposta e em seguida os de teste sem reposta
#
#          | 'Filme' | 'Muito' | 'Bom' | 'Horrivel' | 'Apesar' | 'Elenco' | Resultado
# Treino 1 |    1    |    1    |   1   |      0     |     0    |     0    | Positivo
# Treino 2 |    1    |    0    |   0   |      1     |     1    |     1    | Negativo
# Teste  1 |    1    |    0    |   1   |      1     |     1    |     0    | ???
countvec = CountVectorizer(stop_words = 'english', ngram_range = (1,2), min_df = 2)
treino = countvec.fit_transform(Xtreino.Reviews) # Gera bag of words com palavras que ele nao viu antes
teste = countvec.transform(Xteste.Reviews)
#### Fim do treino
trainingTime = time.time() - trainingStartTime

#### Inicio do algoritimo
runningStartTime = time.time()
# Escolhe algoritimo a ser utilizado
nb = BernoulliNB()
nb = nb.fit(treino, Xtreino.Positive)
# Preve as repostas para o caso de teste
ypred = nb.predict(teste)
#### Fim do algoritimo
runningTime = time.time() - runningStartTime

# Faz a comparacao das respostas obtidas pelo algoritimo com o "gabarito"
# devolve a porcentagem encontrada
acc = accuracy_score(Xteste.Positive, ypred)
accuracy = str(acc)

# Arquivo de saida
file = open("resultados_bernoulli_naive_bayes.txt", "w")
print "Resultado: " + accuracy
print "Tempo de treinamento: " + str(trainingTime) + "s"
print "Tempo de classificacao: " + str(runningTime) + "s"

file.write("Bernoulli Naive Bayes:")
file.write("\nResultado: " + accuracy)
file.write("\nTempo de treinamento: " + str(trainingTime) + "s")
file.write("\nTempo de classificacao: " + str(runningTime) + "s")
file.close()