# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

Algorithm:

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


Program and Output:

/*
Program to implement univariate Linear Regression to fit a straight line using least squares.

Developed by: SIVA SAKTHI A

RegisterNumber: 212224223005
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('drive/MyDrive/Dataset-ML/student_scores.csv')
df.head()
```
```
df.tail()
```
```
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
X,Y
```
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
Y_pred,Y_test
```
```
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Hours VS Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title('Hours VS Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
```
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("MSE = ",mse)
print('MAE = ',mae)
print('RMSE = ',rmse)
```
OUTPUT:

<img width="578" height="317" alt="image" src="https://github.com/user-attachments/assets/a2deac61-d355-43a9-8e3c-746ceb383ff4" />
<img width="387" height="267" alt="image" src="https://github.com/user-attachments/assets/cad869dd-66d8-4d3f-817c-2bdfed4edf71" />
<img width="802" height="593" alt="image" src="https://github.com/user-attachments/assets/bbd2cf92-e209-473b-a44b-315b851b6103" />
<img width="782" height="87" alt="image" src="https://github.com/user-attachments/assets/e688cfd1-5fb2-44eb-a324-26ffe433a585" />
<img width="803" height="592" alt="image" src="https://github.com/user-attachments/assets/1277b9bb-660a-4a40-9de4-fe801d8b6a5a" />
<img width="762" height="512" alt="image" src="https://github.com/user-attachments/assets/917cc0cf-75b3-4d3b-8cc1-09a8a6356d51" />
<img width="772" height="85" alt="image" src="https://github.com/user-attachments/assets/c50c57f6-4596-4710-af14-b6f3f3c8e01c" />


Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
