import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score as evs

from sklearn.ensemble import RandomForestRegressor


# Data Import
df = pd.read_csv('student-por.csv')


# EDA & Data Preprocessing
print(df)

print(df.isnull().sum())


for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

print(df.info())

print(df.dtypes)


# Train Test Split
X = df.drop('Dalc', axis= 1)
Y = df['Dalc']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    shuffle= True,
                                                    random_state= 24,
                                                    test_size= 0.20
                                                    ) 


# Model Training
m = RandomForestRegressor()

m.fit(X_train, Y_train)


# Model Training & Testing Accuracy
print(m, '\n')

pred_train = m.predict(X_train)
print(f'Train Accuracuracy is :{evs(Y_train, pred_train)}')

pred_test = m.predict(X_test)
print(f'Test Accuracy is :{evs(Y_test, pred_test)}')