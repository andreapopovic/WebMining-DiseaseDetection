import streamlit as strl
from streamlit_option_menu import option_menu
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
with strl.sidebar:
    selected = option_menu('Disease Prediction System', 
                           ['Heart Disease','Stroke','Diabetes'],
                           icons = ['heart','person','activity'],
                           default_index=0)
if(selected == 'Heart Disease'):
    strl.title('Heart Disease prediction')   

    Age = strl.number_input('Age of the person')
    Sex = strl.number_input('Gender of the person: 0-female, 1-male')
    ChestPain = strl.number_input('Type of chest-pain: 1 = typical angina, 2 = atypical angina, 3 = non — angina pain, 4 = asymptotic')
    BloodPressure = strl.number_input('Blood pressure value in mmHg ')
    SerumCholesterol = strl.number_input('Serum cholesterol in mg/dl')
    FastingBloodSugar = strl.number_input('If fasting blood sugar > 120mg/dl then : 1, else : 0 ')
    ECG = strl.number_input('Resting electrocardiographic results: 0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hypertrophy')
    HeartRate = strl.number_input('Max heart rate ')
    Angina = strl.number_input('Exercise induced angina : 1 = yes, 0 = no ')
    STDepression = strl.number_input('ST depression induced by exercise relative to rest: integer or float number')
    STSegment = strl.number_input('Peak exercise ST segment : 1 = upsloping, 2 = flat ,3 = downsloping')
    Vesseles = strl.number_input('Number of major vessels (0–3) colored by fluoroscopy: displays the value as integer or float')
    Thal = strl.number_input('Thal : displays the thalassemia : 3 = normal, 6 = fixed defect, 7 = reversible defect')
    
    if strl.button('Heart Desease Test Result'):
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
        parameters = [[Age, Sex, ChestPain, BloodPressure, SerumCholesterol, FastingBloodSugar, ECG, HeartRate,Angina, STDepression, STSegment, Vesseles, Thal]]
        parameters = st_x.transform(parameters)  
        prediction = classifier.predict(parameters)
        

        if (prediction[0] == 1):
          diagnosis = 'The person has a high chance of having a heart attack. '
        else:
          diagnosis = 'The person does not have a high chance of having a heart attack. ' 
        strl.success(diagnosis)


if(selected == 'Stroke'):
    strl.title('Stroke prediction') 
    dataset = pd.read_csv('dataset/stroke.csv')
    dataset[dataset['bmi'].isnull()]['smoking_status'].value_counts()
    dataset['bmi'] = dataset['bmi'].interpolate(method ='linear', limit_direction='forward')
    dataset.drop(['id'], axis=1, inplace=True)
    dataset['gender'] = dataset['gender'].replace(['Male'], 0.0)
    dataset['gender'] = dataset['gender'].replace(['Female'], 1.0)
    dataset['gender'] = dataset['gender'].replace(['Other'], 2.0)
    dataset['ever_married'] = dataset['ever_married'].replace(['No'], 0.0)
    dataset['ever_married'] = dataset['ever_married'].replace(['Yes'], 1.0)
    dataset['work_type'] = dataset['work_type'].replace(['Private'], 0.0)
    dataset['work_type'] = dataset['work_type'].replace(['Self-employed'], 1.0)
    dataset['work_type'] = dataset['work_type'].replace(['Govt_job'], 2.0)
    dataset['work_type'] = dataset['work_type'].replace(['children'], 3.0)
    dataset['work_type'] = dataset['work_type'].replace(['Never_worked'], 4.0)
    dataset['Residence_type'] = dataset['Residence_type'].replace(['Urban'], 0.0)
    dataset['Residence_type'] = dataset['Residence_type'].replace(['Rural'], 1.0)
    dataset['smoking_status'] = dataset['smoking_status'].replace(['Unknown'], 0.0)
    dataset['smoking_status'] = dataset['smoking_status'].replace(['smokes'], 1.0)
    dataset['smoking_status'] = dataset['smoking_status'].replace(['formerly smoked'], 2.0)
    dataset['smoking_status'] = dataset['smoking_status'].replace(['never smoked'], 3.0)


    Gender = strl.number_input('Gender of the person: 0-female, 1-male, 2-other')
    Age = strl.number_input('Age of the person')
    Hypertension = strl.number_input('Have you ever had hypertension? : 0 = No, 1 = Yes')
    HeartDisease = strl.number_input('Have you ever had heart disease? : 0 = No, 1 = Yes')
    Married = strl.number_input('Are you married? : 0 = No, 1 = Yes ')
    WorkType = strl.number_input('Work type? 0 = Private, 1 = Self-employed, 2 = Children, 3 = Never_worked ')
    Residence = strl.number_input('Residence type? 0 = Urban, 1 = Rural')
    Glucose = strl.number_input('Average glucose level')
    BMI = strl.number_input('BMI value')
    SmokingStatus = strl.number_input('SmokingStatus? 0 = Unknown, 1 = Smokes, 2 = Formerly smoked, 3 = Never smoked ')
    parameters = [[Gender, Age, Hypertension, HeartDisease, Married, WorkType, Residence, Glucose, BMI, SmokingStatus]]
    if strl.button('Stroke Test Result'):
      x = dataset.drop(['stroke'], axis=1).values
      y = dataset['stroke'].values
      st_x = StandardScaler()    
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
      x_train=st_x.fit_transform(x_train)
      parameters=st_x.transform(parameters)
      rf=RandomForestClassifier()
      rf.fit(x_train,y_train)
      prediction=rf.predict(parameters)
      
      
      if (prediction[0] == 1):
          diab_diagnosis = 'The person has a high chance of having a stroke.'
      else:
          diab_diagnosis = 'The person has not a high chance of having a stroke.'
      strl.success(diab_diagnosis)
      

if(selected == 'Diabetes'):
    strl.title('Diabetes prediction')   
    
    Pregnancies = strl.number_input('Number of Pregnancies')
    Glucose = strl.number_input('Glucose Level')
    BloodPressure = strl.number_input('Blood Pressure value')
    SkinThickness = strl.number_input('Skin Thickness value')
    Insulin = strl.number_input('Insulin Level')
    BMI = strl.number_input('BMI value')
    DiabetesPedigreeFunction = strl.number_input('Diabetes Pedigree Function value')
    Age = strl.number_input('Age of the Person')  
    
    if strl.button('Diabetes Test Result'):
        diabetes_dataset = pd.read_csv('D:/WebMining/dataset/diabetes.csv')
        x= diabetes_dataset.iloc[:,0:8].values 
        y= diabetes_dataset['Outcome'].values
        x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=42)
        DecisionTree = tree.DecisionTreeClassifier(random_state = 42)
        DecisionTree = DecisionTree.fit(x_train, y_train)
        parameters = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,DiabetesPedigreeFunction,Age]]
        prediction = DecisionTree.predict(parameters)
        
        if (prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic.'
        else:
          diab_diagnosis = 'The person is not diabetic.'
        strl.success(diab_diagnosis)