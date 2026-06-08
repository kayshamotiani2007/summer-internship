import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("churn.db")
query = "SELECT * FROM churn_modelling"
df = pd.read_sql(query, conn)

df.head()


print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df['Surname'] = lb.fit_transform(df['Surname'])
df['Geography'] = lb.fit_transform(df['Geography'])
df['Gender'] = lb.fit_transform(df['Gender'])

x = df.drop(columns=['Exited'])
y = df['Exited']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression
lr.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
