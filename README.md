# Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: A.Sharilma.
RegisterNumber: 212221230094.
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://user-images.githubusercontent.com/94882905/203790369-1e504865-18c5-47be-985e-f4051ea52db9.png)

![image](https://user-images.githubusercontent.com/94882905/203790388-6259e90f-26bc-492b-98f9-fd50c49a8782.png)

![image](https://user-images.githubusercontent.com/94882905/203790421-0195db10-6787-4992-8491-655d21efcc82.png)

![image](https://user-images.githubusercontent.com/94882905/203790459-5b140942-c7b6-4751-b2c1-2a5d30e09e2d.png)

![image](https://user-images.githubusercontent.com/94882905/203790517-2e29d94a-c22f-4567-950d-2b34a93e67d0.png)

![image](https://user-images.githubusercontent.com/94882905/203790542-1b9ae100-20da-4498-bd74-0de536df8c98.png)

![image](https://user-images.githubusercontent.com/94882905/203790576-ce61c3ff-9ae7-4eb5-ba09-be5b318d1991.png)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
