#!/usr/bin/env python
# IMPORT PACKAGES
# data handling
# import numpy as np
# import pandas as pd
# import sqlite3
from sklearn.model_selection import train_test_split

# algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# metrics
from sklearn.metrics import fbeta_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# # config
# import os
# import yaml

# utilities
from utils import *


# SET UP CONFIG FILE
# folder to load config file
# CONFIG_PATH = "src"

# # Function to load yaml configuration file
# def load_config(config_name):
#     with open(os.path.join(CONFIG_PATH, config_name)) as file:
#         config = yaml.safe_load(file)

#     return config


# # DATA PROCESSING
# # get data
# # conn = sqlite3.connect('data/survive.db')
# # load data
# df = pd.read_sql_query("SELECT * FROM survive", conn)
# # close connection
# conn.commit()
# conn.close()


# # remove rows with missing data
# df.dropna(axis=0, inplace=True)

# # rename column names
# df.columns = df.columns.str.replace(' ','_')

# # remove redundant columns
# df.drop(config["drop_columns"], axis=1, inplace=True)


# One-Hot Encoding Catergoical Variables
# df['Survive'].replace({"No": 0, "Yes": 1}, inplace=True)
# df = df.astype({'Survive': int})

# df['Gender'].replace({"Female": 0, "Male": 1}, inplace=True)

# df['Smoke'].replace(to_replace="YES", value='Yes', inplace=True)
# df['Smoke'].replace(to_replace="NO", value='No', inplace=True)
# df['Smoke'].replace(to_replace="Yes", value=1, inplace=True)
# df['Smoke'].replace(to_replace="No", value=0, inplace=True)

# df['Ejection_Fraction'].replace(to_replace="L", value='Low', inplace=True)
# df['Ejection_Fraction'].replace(to_replace="N", value='Normal', inplace=True)
# df['Ejection_Fraction'].replace(to_replace="Low", value=1, inplace=True)
# df['Ejection_Fraction'].replace(to_replace="Normal", value=0, inplace=True)

# df = pd.get_dummies(df, columns=['Diabetes'])


# # replace Ages with negative values
# df['Age'] = np.abs(df['Age'])

# get obesity
# df['Height'] = df['Height']/100
# df['bmi'] = df['Weight'] / (df['Height'] * df['Height'])
# df['obesity'] = np.where(df['bmi'] >25 , 1, 0)
# # remove columns from df
# df.drop(['Height', 'Weight', 'bmi'], axis=1, inplace=True)


# # remove suspected synthetic data
# df = df[df['Ejection_Fraction'] !='High']


# SET UP CONFIG FILE
config = load_config("config.yaml")

# DATA PROCESSING
df = get_data("../data/survive.db") # get_data("data/survive.db")
df = clean_data(df)

# One-Hot Encoding Catergoical Variables
columns = ['Survive', 'Smoke']
dummy_columns = ['Diabetes']
df = encoding_cat_cols(df, columns, dummy_columns)

# get obesity
df = get_obesity_indicator(df, 'Height', 'Weight')

# Normalising Numerical Variables
cols_nor = config["normal_columns"]
df[cols_nor] = (df[cols_nor]-df[cols_nor].mean())/df[cols_nor].std()


# MODELLING
# create independent and depedendent variables
target = df[config["target_name"]]
features = df.copy().drop(config["target_name"], axis=1)


# split data into train and test set
# improvement #5
# catch errors and exceptions during execution 
check_lst = df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).values

if False in check_lst:
    raise TypeError('Inputs must be an int or float')

try:
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config["test_size"], random_state=42)
except TypeError:
    print('Inputs must be an int or float')

print("X train shape: ", x_train.shape)
print("X test shape: ", x_test.shape)
print("Y train shape: ", y_train.shape)
print("Y test shape: ", y_test.shape)


# model selector
# kNN
if config["selected_model"] == "kNN":
    clf = KNeighborsClassifier(
        n_neighbors=config["n_neighbors"],
        weights=config["weights"],
        algorithm=config["algorithm"],
        leaf_size=config["leaf_size"],
        p=config["p"],
        metric=config["metric"],
        n_jobs=config["n_jobs"],
    )
# Rf
elif config["selected_model"] == "Rf":
    clf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        criterion=config["criterion"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        min_weight_fraction_leaf=config["min_weight_fraction_leaf"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        oob_score=config["oob_score"],
        n_jobs=config["n_jobs"],
        random_state=config["random_state"],
        verbose=config["verbose"],
        class_weight=config["class_weight"])
# SVC
elif config["selected_model"] == "SVC":
    clf = SVC(
        C=config["C"],
        kernel=config["kernel"],
        degree=config["degree"],
        gamma=config["gamma"],
        probability=config["probability"],
        class_weight=config["class_weight"],
        verbose=config["verbose_svc"],
        max_iter=config["max_iter"],
        decision_function_shape=config["decision_function_shape"],
        break_ties=config["break_ties"],
        random_state=config["random_state"])
# error
else:
    print("Please check the configuration settings for selected_model.")
    print("Accepted inputs are (case sensitive): kNN, Rf and SVC.")

# train model 
clf.fit(x_train, y_train)

# make predictions
y_pred = clf.predict(x_test)


# EVALUATION
# show current classifier
print("Results for : " + str(clf))

# determine F2 for test set
f2_score = fbeta_score(y_test, y_pred, beta=2)
print(f"F2 SCORE: {f2_score}")

# determine Kappa Score for test set
kappa_score = cohen_kappa_score(y_test, y_pred)
print(f"KAPPA SCORE: {kappa_score}")

# show confusion matrix
matrix = confusion_matrix(y_test,y_pred, labels=[1,0])
print('Confusion matrix : \n',matrix)
matrix = classification_report(y_test,y_pred,labels=[1,0])
print('Classification report : \n',matrix)