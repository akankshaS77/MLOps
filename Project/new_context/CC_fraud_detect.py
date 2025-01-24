
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

df=pd.read_csv('CC_Dataset.xls')
df.head()


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

# accuracy score
dt_score = accuracy_score(y_pred, y_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)
print('Accuracy score of logistic regression is:', lr_score)

conf_matrix = confusion_matrix(y_test, y_pred1)
print(conf_matrix)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.3, shuffle=True)


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# accuracy score
dt_score = accuracy_score(y_pred, y_test)


conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)
print('Accuracy score of logistic regression is:', lr_score)

conf_matrix = confusion_matrix(y_test, y_pred1)
print(conf_matrix)


print(classification_report(y_test, y_pred1))

### Hyperparameters Tuning of Decision Tree algorithm

# %%
dt_hp = DecisionTreeClassifier(random_state=43)

params = {'max_depth':[3,5,7,10,15],
          'min_samples_leaf':[3,5,10,15,20],
          'min_samples_split':[8,10,12,18,20,16],
          'criterion':['gini','entropy']}
GS = GridSearchCV(estimator=dt_hp,param_grid=params,cv=5,n_jobs=-1, verbose=True, scoring='accuracy')

# %%
GS.fit(X_train, y_train)

# %%
print('Best Parameters:',GS.best_params_,end='\n\n')
print('Best Score:',GS.best_score_)

# %%
GS.fit(X_train, y_train)

# %%
y_test_pred = GS.predict(X_test)

class_count_0, class_count_1 = df['Is Fraudulent'].value_counts()
# Separate class
class_0 = df[df['Is Fraudulent'] == 0]
class_1 = df[df['Is Fraudulent'] == 1]

class_0_under = class_0.sample(class_count_1)
test_under = pd.concat([class_0_under, class_1], axis=0)

# %%
test_under

# %%
#X=df_norm
#df.drop(['A'], axis=1)
#y=df['Is Fraudulent'].values
X=test_under.drop(['Is Fraudulent'], axis=1)
y=test_under['Is Fraudulent'].values

# %%
# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.20, shuffle=True)

# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)
print('Accuracy score of logistic regression is:', round(lr_score, 3))

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)
print('Accuracy score of decision tree is:', round(dt_score, 3))

# %% [markdown]
# #### Random Over-Sampling
# Oversampling can be defined as adding more copies to the minority class. Oversampling can be a good choice when you don’t have a ton of data to work with.

# %%
class_1_over = class_1.sample(class_count_0, replace=True)
test_over = pd.concat([class_1_over, class_0], axis=0)
print("total class of 1 and 0:",test_over['Is Fraudulent'].value_counts())# plot the count after under-sampeling
test_over['Is Fraudulent'].value_counts().plot(kind='bar', title='count (target)')

# %%
test_over.info()

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
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)
print('Accuracy score of logistic regression is:', round(lr_score, 3))

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)
print('Accuracy score of decision tree is:', round(dt_score, 3))

# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)
print('Accuracy score of logistic regression is:', round(lr_score, 3))

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)
print('Accuracy score of decision tree is:', round(dt_score, 3))

# %%
##Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
#Predict the response for test dataset
y_pred1 = lr.predict(X_test)
# accuracy score
lr_score = accuracy_score(y_pred1, y_test)
print('Accuracy score of logistic regression is:', round(lr_score, 3))

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)
print('Accuracy score of decision tree is:', round(dt_score, 3))

# %% [markdown]
# #### NearMiss (Undersampling Technique)
# NearMiss is an under-sampling technique. Instead of resampling the Minority class, using a distance will make the majority class equal to the minority class.

# %%
from imblearn.under_sampling import NearMiss

nm = NearMiss()

x_nm, y_nm = nm.fit_resample(X, y)

print('Original dataset shape:', Counter(y))
print('Resample dataset shape:', Counter(y_nm))

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
print('Accuracy score of logistic regression is:', round(lr_score, 3))

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# accuracy score
dt_score = accuracy_score(y_pred, y_test)
print('Accuracy score of decision tree is:', round(dt_score, 3))


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

# %% [markdown]
# ####  Resampling techniques are often used in combination with cross-validation, especially when dealing with imbalanced datasets. Here, we use SMOTE technique also used in Assignment 1 above assisting us in generating synthetic samples for minority classes which helps adequately in improving model performance and addressing class imbalance issues.

# %%
# Call the build_models function to train the models
knn_model, nb_model, rf_model, adaboost_model = build_models(X_train, y_train)

# %% [markdown]
# ### Performance Evaluation

# %%
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

# %%
# Call the evaluate_models function with the trained models dictionary
evaluation_results, confusion_matrices = evaluate_models(trained_models, X_test, y_test)

# Print the Confusion Matrix
print('Confusion Matrix: \n')
for name, conf_mat in confusion_matrices.items():
    print(f'{name} : {conf_mat}\n')

# Print the evaluation results DataFrame
print("Evaluation Results:")
print(f'{evaluation_results}\n')


for name, conf_mat in confusion_matrices.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, cmap='coolwarm', fmt='d')  # Modified fmt parameter
    fig.suptitle(t=f"Confusion Matrix - {name}",
                 color="orange",
                 fontsize=16)
    ax.set(xlabel="Predicted Label",
           ylabel="Actual Label")
    plt.show()


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

best_hyperparameters = hyperparameter_tuning(X_train, y_train)

print("Best Hyperparameters:")
for model, params in best_hyperparameters.items():
    print(model, ":", params)

# %%
eval_results, conf_matr_ = evaluate_models(trained_models, X_test, y_test)

# Print the Confusion Matrix
print('Confusion Matrix: \n')
for name, conf_mat in conf_matr_.items():
    print(f'{name} : {conf_mat}\n')

# Print the evaluation results DataFrame
print("Evaluation Results:")
print(evaluation_results)


for name, conf_mat in conf_matr_.items():
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(conf_mat, annot=True, cmap='coolwarm', fmt='g')
    fig.suptitle(t=f"Confusion Matrix - {name}",
                    color="orange",
                    fontsize=16)
    ax.set(xlabel="Predicted Label",
            ylabel="Actual Label")
    plt.show()

import joblib   
joblib.dump(rf_model,'rf_model')


