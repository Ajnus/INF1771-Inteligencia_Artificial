import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from random import randint

## funcao que gera numeros impares, passando como parametros: o comeco do intervalo do conjunto, o final e quantidade de numeros impares
def generateOddNumbers(start, end, qtdOfOdds):
    Odds = []
    while (qtdOfOdds > 0):
        randNumber = randint(start, end)
        if(randNumber % 2 == 0):
            randNumber = randNumber + 1
            Odds.append(randNumber)
            qtdOfOdds = qtdOfOdds - 1
            print(randNumber)
    return Odds

Xtreino = pd.read_csv('treino.csv')
Xteste = pd.read_csv('teste.csv')

countvec = CountVectorizer(stop_words = 'english', ngram_range = (1,2), min_df = 2)

treino = countvec.fit_transform(Xtreino.Reviews) # Gera bag of words com palavras que ele nao viu antes

teste = countvec.transform(Xteste.Reviews)

Odds = generateOddNumbers(260, 290, 30)
size = len(Odds)
sizeNeigh = len(Odds)
accMax = 0
kMax = 0
while(size > 0):
    file = open("resultados_3.txt", "a")
    k = Odds.pop()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(treino,Xtreino.Positive)

    ypred = knn.predict(teste)

    acc = accuracy_score(Xteste.Positive, ypred)

    if(acc > accMax):
        kMax = k
        accMax = acc
    k = str(k)
    acc = str(acc)
    print "Resultado: " + k + " - " + acc
    size = size - 1
    resultado = "K: " + k + " - " + acc + "\n"
    file.write(resultado)
    file.close()

file = open("resultados_3.txt", "a")
resultadoMax = "Tamanho da Vizinhanca:  " + str(sizeNeigh)+ " " + "K: " + str(kMax)+ " " + "Acc: " + str(accMax) + "\n"
file.write(resultadoMax)
file.close()
