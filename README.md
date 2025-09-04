# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1:
Import the standard libraries such as pandas module to read the corresponding csv file.

### Step 2:
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

### Step 3:
Import LabelEncoder and encode the corresponding dataset values.

### Step 4:
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

### Step 5:
Predict the values of array using the variable y_pred.

### Step 6:
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

### Step 7:
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

### Step 8:
End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: CHARUKESH S
RegisterNumber:  212224230044
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
print("Name: CHARUKESH S")
print("Register No: 212224230044")
data1.head()
```
```
print("Name: CHARUKESH S")
print("Register No: 212224230044")
data1.isnull().sum()
```
```
print("Name: CHARUKESH S")
print("Register No: 212224230044")
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
print("Name: CHARUKESH S")
print("Register No: 212224230044")
data1
```
```
print("Name: CHARUKESH S")
print("Register No: 212224230044")
x = data1.iloc[:,:-1]
x
```
```
print("Name: CHARUKESH S")
print("Register No: 212224230044")
y = data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print("Name: CHARUKESH S")
print("Register No: 212224230044")
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Name: CHARUKESH S")
print("Register No: 212224230044")
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Name: CHARUKESH S")
print("Register No: 212224230044")
confusion
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Name: CHARUKESH S")
print("Register No: 212224230044")
classification_report1
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### HEAD OF THE DATA:
<img width="955" height="233" alt="image" src="https://github.com/user-attachments/assets/893d115e-2b32-4bc7-b0fc-dcf1efae2b20" />

### SUM OD ISNULL DATA:
<img width="419" height="302" alt="image" src="https://github.com/user-attachments/assets/bb24abbd-e0b9-4b27-a280-3485b162540f" />

### SUM OF DUPLICATE DATA:
<img width="616" height="71" alt="image" src="https://github.com/user-attachments/assets/4ba00803-2ea5-4a4f-9884-7ec1041493c9" />

### LABEL ENCODER OD DATA:
<img width="902" height="414" alt="image" src="https://github.com/user-attachments/assets/3dec03e0-4360-4a3c-b8b4-15cfd218c3e4" />

### ILOC OF X:
<img width="830" height="400" alt="image" src="https://github.com/user-attachments/assets/4a769ece-6183-464d-a7a5-3f84d94c0ce6" />

<img width="539" height="260" alt="image" src="https://github.com/user-attachments/assets/1b404b09-955d-4026-9eab-8307e3603735" />

### PREDICTED VALUES:
<img width="720" height="113" alt="image" src="https://github.com/user-attachments/assets/be0aceeb-0824-4347-950b-9d2f1f61df19" />

### ACCURACY:
<img width="746" height="91" alt="image" src="https://github.com/user-attachments/assets/ecee1fdc-4b6d-4773-a53f-42b12c1599fb" />

### CONFUSION MATRIX:
<img width="604" height="108" alt="image" src="https://github.com/user-attachments/assets/b7eacfa5-7dc9-4b9c-936c-261187222e56" />

### CLASSIFICATION REPORT:
<img width="1085" height="123" alt="image" src="https://github.com/user-attachments/assets/b4aecd72-cda8-4565-b302-6b3af7f7f2dc" />

### PREDICTED LR VALUE:
<img width="809" height="37" alt="image" src="https://github.com/user-attachments/assets/daea6d0b-4647-44dd-ba0d-cb66f4732efe" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
