import string
import os
import pandas as pd

# Description: Carrega os nomes das reviews contidos num diretorio
# param: part: 1 para conjunto de treino, 2 para conjunto teste
# param: isPositive: True se eh do conjunto positivo ou False do negativo
# returns: lista contendo strings de nomes dos arquivos
def loadFileNames(part,isPositive):
    partString = "part" + str(part)
    groupString = "neg"
    if isPositive:
        groupString = "pos"

    cwd = os.getcwd()
    fileList = list(os.listdir(cwd + "/movie_review_dataset/" + partString + "/" + groupString))

    return fileList

# Description: Carrega uma review e remove quebra de linhas por paragrafos
# param: part: 1 para conjunto de treino, 2 para conjunto teste
# param: isPositive: True se eh do conjunto positivo ou False do negativo
# param: filename: nome do arquivo contendo a review
# returns: texto corrido da review (string)
def loadReview(part, isPositive, filename):
    partString = "part" + str(part)
    groupString = "neg"
    if isPositive:
        groupString = "pos"

    cwd = os.getcwd()
    file = open(cwd + "/movie_review_dataset/"+partString+"/"+groupString+"/"+filename)
    fullreview = ""
    for line in file:
        fullreview += (line + " ")

    return fullreview

# Description: Load review list and postive or negative list
# param: part: 1 = conjunto de treino 2 = conjunto de teste
# Returns (reviewList, negpos)
# reviewList: contains the text of the reviews
# negpos: contaisn a list o the same size as the reviewList
# with True if the respective review is positive or False if it is negative
def loadReviewList(part):
    print "Loading file names..."
    posFileList = loadFileNames(part, True) # Get positive reviews from part 1
    negFileList = loadFileNames(part, False) # Get negative reviews from part 1
    print "File names loaded!"

    reviewList = []
    negpos=[] #True positive False Negative
    value = 0
    for name in posFileList:
        review = loadReview(part, True, name)
        reviewList.append(review)
        negpos.append(True)
        value += 1
    print "Numero de review positivas carregadas: " + str(value)
    value = 0
    for name in negFileList:
        review = loadReview(part, False, name)
        reviewList.append(review)
        negpos.append(False)
        value += 1
    print "Numero de review negativas carregadas: " + str(value)

    return (reviewList, negpos)

# Carrega reviews positivas e negativas
(reviewList, isPositiveList) = loadReviewList(1)
(reviewListTest, isPositiveListTest) = loadReviewList(2)

# Gera um data frame da matriz no modelo
#                                              Reviews  Positive
# 0  Based on an actual story, John Boorman shows t...      True
# 1  This is a gem. As a Film Four production - the...      True
# 2  I really like this show. It has drama, romance...      True
# 3  This is the best 3-D experience Disney has at ...      True
# 4  Of the Korean movies I've seen, only three had...      True
print "Numero total de reviews carregadas: " + str(len(reviewList))
Xtreino = pd.DataFrame()
Xteste = pd.DataFrame()

Xtreino['Reviews'] = pd.Series(reviewList)
Xtreino['Positive'] = pd.Series(isPositiveList)

Xteste['Reviews'] = pd.Series(reviewListTest)
Xteste['Positive'] = pd.Series(isPositiveListTest)

# Gera arquivos de CSV para salvar formato da matriz
Xteste.to_csv('teste.csv', index = False)
Xtreino.to_csv('treino.csv', index = False)
