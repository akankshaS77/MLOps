
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

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.20, shuffle=True)

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# accuracy score
dt_score = accuracy_score(y_pred, y_test)

conf_matrix = confusion_matrix(y_test, y_pred)


lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)

conf_matrix = confusion_matrix(y_test, y_pred1)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.25, shuffle=True)


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.3, shuffle=True)


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# accuracy score
dt_score = accuracy_score(y_pred, y_test)


lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)



dt_hp = DecisionTreeClassifier(random_state=43)

params = {'max_depth':[3,5,7,10,15],
          'min_samples_leaf':[3,5,10,15,20],
          'min_samples_split':[8,10,12,18,20,16],
          'criterion':['gini','entropy']}

GS = GridSearchCV(estimator=dt_hp,param_grid=params,cv=5,n_jobs=-1, verbose=True, scoring='accuracy')

# %%
GS.fit(X_train, y_train)

y_test_pred = GS.predict(X_test)

class_count_0, class_count_1 = df['Is Fraudulent'].value_counts()
# Separate class
class_0 = df[df['Is Fraudulent'] == 0]
class_1 = df[df['Is Fraudulent'] == 1]

class_0_under = class_0.sample(class_count_1)
test_under = pd.concat([class_0_under, class_1], axis=0)


X=test_under.drop(['Is Fraudulent'], axis=1)
y=test_under['Is Fraudulent'].values


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.20, shuffle=True)

# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)


clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)
print('Accuracy score of decision tree is:', round(dt_score, 3))

class_1_over = class_1.sample(class_count_0, replace=True)
test_over = pd.concat([class_1_over, class_0], axis=0)
print("total class of 1 and 0:",test_over['Is Fraudulent'].value_counts())# plot the count after under-sampeling
test_over['Is Fraudulent'].value_counts().plot(kind='bar', title='count (target)')


# %%
X=test_over.drop(['Is Fraudulent'], axis=1)
y=test_over['Is Fraudulent'].values

# %%
X.info()

# %%
len(y)

# %%
# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.20)

# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)


# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score


# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)


# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score


# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score


from imblearn.under_sampling import NearMiss

nm = NearMiss()

x_nm, y_nm = nm.fit_resample(X, y)


# %%
# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(x_nm,y_nm,random_state=104,test_size=0.20)

# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)


# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def build_models(X_train, y_train):
    knn_model = KNeighborsClassifier()
    nb_model = GaussianNB()
    rf_model = RandomForestClassifier()
    adaboost_model = AdaBoostClassifier()

    knn_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    adaboost_model.fit(X_train, y_train)

    return knn_model, nb_model, rf_model, adaboost_model


knn_model, nb_model, rf_model, adaboost_model = build_models(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_models(models, X_test, y_test):
    evaluation_results = {}
    confusion_matrices = {}  # Create a dictionary to store confusion matrices
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = conf  # Store confusion matrix for each model

        # Calculate misclassification rate
        misclassification_rate = 1 - accuracy

        # Create a dictionary to store evaluation metrics for each model
        evaluation_results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Misclassification Rate': misclassification_rate
        }

    # Convert the dictionary of evaluation metrics into a DataFrame
    metrics_df = pd.DataFrame(evaluation_results).T

    return metrics_df, confusion_matrices

# %%
# Define trained models in a dictionary
trained_models = {
    'KNN': knn_model,
    'Naive Bayes': nb_model,
    'Random Forest': rf_model,
    'Adaboost': adaboost_model
}

evaluation_results, confusion_matrices = evaluate_models(trained_models, X_test, y_test)


# Print the evaluation results DataFrame
print("Evaluation Results:")
print(f'{evaluation_results}\n')



from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(X_train, y_train):
    # Define parameter grids for each classifier
    knn_param_grid = {'n_neighbors': [1,3, 5, 7, 9]}
    rf_param_grid = {'n_estimators': [50, 100,150,200,250], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 15]}
    adaboost_param_grid = {'n_estimators': [50, 100, 200,250], 'learning_rate': [0.01, 0.1, 1,1.5]}

    # Define classifiers and their respective parameter grids
    classifiers = {
        'KNN': (KNeighborsClassifier(), knn_param_grid),
        'Random Forest': (RandomForestClassifier(), rf_param_grid),
        'Adaboost': (AdaBoostClassifier(), adaboost_param_grid),
        'Naive Bayes': (GaussianNB(), {})  # No hyperparameters for Naive Bayes
    }

    # Perform hyperparameter tuning for each classifier
    best_params = {}
    for name, (model, param_grid) in classifiers.items():
        if param_grid:  # Check if there are hyperparameters to tune
            if name == 'Adaboost':
                model = AdaBoostClassifier(algorithm='SAMME')  # Explicitly set algorithm to 'SAMME'
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
            grid_search.fit(X_train, y_train)
            best_params[name] = grid_search.best_params_
        else:
            best_params[name] = "No hyperparameters to tune for Naive Bayes"

    return best_params

with mlflow.start_run():
    #log best hyperparameters
    best_hyperparameters = hyperparameter_tuning(X_train, y_train)
    print("Best Hyperparameters:")
    for model, params in best_hyperparameters.items():
        if isinstance(params, dict):  # If parameters are a dictionary, log each key-value pair
            for param_name, value in params.items():
                mlflow.log_param(f"{model}_{param_name}", value)
        else:
            mlflow.log_param(f"{model}_params", params)

   
    for model, params in best_hyperparameters.items():
        print(model, ":", params)

    # Log Evaluation meterics
    eval_results, conf_matr_ = evaluate_models(trained_models, X_test, y_test)
    for model, metrics in eval_results.iterrows():
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model}_{metric_name}", value)

    #Log Confusion matrices as artifacts
    import os

    for model_name, conf_matrix in conf_matr_.items():
        conf_matrix_path = f"{model_name}_confusion_matrix.csv"
        pd.DataFrame(conf_matrix).to_csv(conf_matrix_path, index=False)
        mlflow.log_artifact(conf_matrix_path)
        os.remove(conf_matrix_path)  # Clean up after logging

    
    #Log Model
    for model_name, model_obj in trained_models.items():
         mlflow.sklearn.log_model(model_obj, model_name)
    
    


