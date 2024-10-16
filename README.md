# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Attach the given data file
3. Now find the satisfaction level of employee data
4. Find the accuracy and new predict value
5. End the program


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SADHANA SHREE B
RegisterNumber: 212223230177
*/
import pandas as pd
data=pd.read_csv("Employee (1).csv")
data.head()
data.tail()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,5,2]])

```

## Output:


![Screenshot 2024-10-16 112513](https://github.com/user-attachments/assets/a7064d5a-afb1-4533-b833-1ae7aa0c7cd2)


![Screenshot 2024-10-16 112519](https://github.com/user-attachments/assets/623f05da-9696-4e20-aaf6-6ad8cf55fe5d)


![Screenshot 2024-10-16 112524](https://github.com/user-attachments/assets/b54d39ec-c073-45ad-8307-498645c8f70d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
