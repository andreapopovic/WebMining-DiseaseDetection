import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


heart_disease_dataset = pd.read_csv('D:/WebMining/dataset/heart.csv')
x= heart_disease_dataset.iloc[:,0:13].values 
y= heart_disease_dataset['target'].values
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
st_x = StandardScaler()    
x_train = st_x.fit_transform(x_train)    
x_test = st_x.transform(x_test)  
error = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
K = error.index(min(error))+1
classifier = KNeighborsClassifier(n_neighbors = K)  
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
accuracy = accuracy_score(y_test,prediction)
print('Accuraci score of prediction is ',accuracy,'%.')
print(classification_report(y_test,prediction))