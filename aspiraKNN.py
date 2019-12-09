import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Read the Training and Testing Data:
data_train = pd.read_csv(filepath_or_buffer="poker-hand-training-true.data", sep=',', header=None)
data_test = pd.read_csv(filepath_or_buffer="poker-hand-testing.data", sep=',', header=None)

#Print it's Shape to get an idea of the data set:
print(data_train.shape)
print(data_test.shape)

#Prepare the Data for Training and Testing:
#Ready the Train Data
array_train = data_train.values
data_train = array_train[:,0:10]
label_train = array_train[:,10]
#Ready the Test Data
array_test = data_test.values
data_test = array_test[:,0:10]
label_test = array_test[:,10]

trainingStartTime = time.time()
'''
# Scaling the Data for our Main Model
# Scale the Data to Make the NN easier to converge
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(data_train)  
# Transform the training and testing data
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
'''
# Init the Models for Comparision
model = KNeighborsClassifier(n_neighbors = 5)
name = "KNN"

model.fit(data_train, label_train)
trainingTime = time.time() - trainingStartTime

runningStartTime = time.time()
#Predict
prediction = model.predict(data_test)
runningTime = time.time() - runningStartTime

# Print Accuracy
acc = accuracy_score(label_test, prediction)
print("Accuracy Using",name,": " + str(acc)+'\n')
print("Tempo de treinamento: " + str(trainingTime) + ' segundos')
print("Tempo de classificação: " + str(runningTime) + ' segundos')    