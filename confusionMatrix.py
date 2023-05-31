
import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay



diabetes_dataset = pd.read_csv('D:/WebMining/dataset/diabetes.csv')
x= diabetes_dataset.iloc[:,0:8].values 
y= diabetes_dataset['Outcome'].values
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=42)
DecisionTree = tree.DecisionTreeClassifier(random_state = 42)
DecisionTree = DecisionTree.fit(x_train, y_train)
prediction = DecisionTree.predict(x_test)
cm = confusion_matrix(y_test,prediction,labels=DecisionTree.classes_)
displ = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=DecisionTree.classes_)
displ.plot()
plt.show()