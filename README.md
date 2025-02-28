# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
### Developed by:SARWESHVARAN A
### RegisterNumber:212223230198  

```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(*X)
Y=df.iloc[:,1].values
print(*Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
print(*Y_pred)
Y_test
print(*Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```




## Output:
```
df.head()
```
![image](https://github.com/user-attachments/assets/d84a43de-a91f-4c7d-9a39-4f5640ddac08)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/f2c02426-049e-48f2-9b34-37a2d69a1d25)
```
values of X
```
![image](https://github.com/user-attachments/assets/89e93cbe-a344-4884-b392-7de42ef4883f)
```
values of y
```
![image](https://github.com/user-attachments/assets/54baac72-f3e0-4354-bb31-16347e356800)
```
y_pred
```
![image](https://github.com/user-attachments/assets/aa64793c-c4e7-4921-80f0-fbe8be798268)
```
y_test
```
![image](https://github.com/user-attachments/assets/fbc6e125-2e4d-499d-a164-66a4e7b1a81d)
```
Training Set
```
![image](https://github.com/user-attachments/assets/e1bb1fc3-f91e-4c4e-a6b4-2a45adb77758)
```
Test Set
```
![image](https://github.com/user-attachments/assets/dad9e8a6-5b6c-4ef7-86a0-1de2826c92c5)
```
Mse,Mae,Rsme
```
![image](https://github.com/user-attachments/assets/1e846c9c-385d-4728-a890-578126018c57)












## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
