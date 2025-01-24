import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import *
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored as style # for text customization
from pandas import DataFrame
import mlflow
import mlflow.sklearn

df=pd.read_csv('C:\\Users\\Aorus\\OneDrive\\Desktop\\MLOps\\MLOps_2\\MLOps\\data\\CC_Dataset.xls')
df.head()
mlflow.set_experiment("CC Fraud Detection")

df.drop(['Unnamed: 0'], axis=1,inplace=True)

df.groupby(by=['Is Fraudulent']).count()
df['Is Fraudulent'].unique()
fraud=df[df['Is Fraudulent']==1]
normal=df[df['Is Fraudulent']==0]
fraud_frc = len(fraud)/float(len(df))


df['Date'] = pd.to_datetime(df['Date'])


df.dtypes   # It will print the data types of all columns (to check datatype for date column)
filtered_df=df.select_dtypes(include=np.number)

filtered_df.columns

correlation_matrix = filtered_df.corr()


df.isnull().sum()

filtered_df.columns
df.groupby(['Card Type'])['Is Fraudulent'].mean()

Mean_encoded = df.groupby(['Card Type'])['Is Fraudulent'].mean().to_dict()
df['Card Type'] = df['Card Type'].map(Mean_encoded)

Mean_encoded = df.groupby(['MCC Category'])['Is Fraudulent'].mean().to_dict()
df['MCC Category'] = df['MCC Category'].map(Mean_encoded)

Mean_encoded = df.groupby(['Device'])['Is Fraudulent'].mean().to_dict()
df['Device'] = df['Device'].map(Mean_encoded)


Mean_encoded = df.groupby(['Location'])['Is Fraudulent'].mean().to_dict()
df['Location'] = df['Location'].map(Mean_encoded)

Mean_encoded = df.groupby(['Merchant Reputation'])['Is Fraudulent'].mean().to_dict()
df['Merchant Reputation'] = df['Merchant Reputation'].map(Mean_encoded)

Mean_encoded = df.groupby(['Online Transactions Frequency'])['Is Fraudulent'].mean().to_dict()
df['Online Transactions Frequency'] = df['Online Transactions Frequency'].map(Mean_encoded)


df.drop('Date', axis=1, inplace=True)

from sklearn.preprocessing import MinMaxScaler
# create a MinMaxScaler object
scaler = MinMaxScaler()

# fit and transform the data
normalized_data = scaler.fit_transform(df.drop('Is Fraudulent', axis=1))

#normalized_data
from pandas import DataFrame
# convert the array back to a dataframe
df_norm = DataFrame(normalized_data)

column_names = list(df.columns)
column_names.pop()

df_norm.columns = column_names

X=df_norm
y=df['Is Fraudulent'].values



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.25, shuffle=True)


from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

with mlflow.start_run(run_name="SMOTE Decision Tree"):
    clf = DecisionTreeClassifier()
    clf.fit(X_smote, y_smote)
    y_pred = clf.predict(X_test)
    mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("F1 Score", f1_score(y_test, y_pred))
    mlflow.sklearn.log_model(clf, "Decision_Tree_SMOTE")
