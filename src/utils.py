# improvement #4
# add docstrings to custom functions

# IMPORT PACKAGES
# data handling
import numpy as np
import pandas as pd
import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns

# config
import os
import yaml


def load_config(config_name, CONFIG_PATH=""):
	""" Function to load yaml configuration file.

	:param config_name: name of targeted yaml file
	:param CONFIG_PATH: path to targeted yaml file
	:return: config dictionary

	>>>  load_config("config.yaml", "src")
	returns config dictionary retrieved from config.yaml in src folder
	"""
	with open(os.path.join(CONFIG_PATH, config_name)) as file:
		config = yaml.safe_load(file)

	return config


def get_data(DB_PATH):
	"""Establish connection with survive.db and save data as DataFrame df.

	:param DB_PATH: path to .db file
	:return: dataframe retrieved from .db file

	>>>  get_data("data/survive.db")
	returns dataframe retrieved from survive.db in data folder
	"""
	conn = sqlite3.connect(DB_PATH)
	# load data
	global df
	df = pd.read_sql_query("SELECT * FROM survive", conn)

	print("data is successfully saved!")
	# close connection
	conn.commit()
	conn.close()

	return df


def clean_data(dataframe, CONFIG_PATH=""):
	"""Perform general data cleaning

	remove rows with missing data
	replace whitespaces in column names with underscore
	remove redundant columns stated in config.yaml
	replace Ages with negative values
	remove suspected synthetic data

	:param dataframe: dataframe to clean
	:param CONFIG_PATH: path to config.yaml
	:return: cleaned dataframe

	>>> clean_data(df)
	df
	"""
	dataframe.dropna(axis=0, inplace=True)
	dataframe.columns = dataframe.columns.str.replace(' ', '_')

	config = load_config("config.yaml", CONFIG_PATH)
	dataframe.drop(config["drop_columns"], axis=1, inplace=True)

	dataframe['Age'] = np.abs(dataframe['Age'])
	dataframe = dataframe[dataframe['Ejection_Fraction'] != 'High']

	return dataframe


def encoding_cat_cols(dataframe, columns, dummy_columns=[]):
	"""Encode all categorical columns.

	:param dataframe: dataframe containing categorical columns to encode
	:param columns: binary categorical columns to encode, can be a list containing multiple column names
	:param dummy_columns: non-binary categorical columns to encode with get_dummies(), can be a list containing multiple column names
	:return: dataframe with encoded categorical columns

	>>> encoding_cat_cols(df, ['Survive', 'Smoke'], ['Diabetes'])
	df
	"""
	# convert all words into lowercase
	dataframe[columns] = dataframe[columns].apply(lambda x: x.astype(str).str.lower())

	# encode yes to 1, no to 0
	dataframe[columns] = dataframe[columns].replace({"no": 0, "yes": 1})
	dataframe[columns] = dataframe[columns].astype(int)

	dataframe['Gender'].replace({"Female": 0, "Male": 1}, inplace=True)

	dataframe['Ejection_Fraction'].replace(to_replace="L", value='Low', inplace=True)
	dataframe['Ejection_Fraction'].replace(to_replace="N", value='Normal', inplace=True)
	dataframe['Ejection_Fraction'].replace(to_replace="Low", value=1, inplace=True)
	dataframe['Ejection_Fraction'].replace(to_replace="Normal", value=0, inplace=True)

	# convert non-binary columns if any
	if dummy_columns:
		dataframe = pd.get_dummies(dataframe, columns=dummy_columns)

	return dataframe


def get_obesity_indicator(dataframe, height_col, weight_col):
	"""Create encoded obesity indicator column.

	:param dataframe: dataframe containing height and weight columns
	:param height_col: name of column containing height in cm
	:param weight_col: name of column containing weight in kg
	:return: dataframe with newly-created obesity column, both height_col and weight_col removed

	>>> get_obesity_indicator(df, 'Height', 'Weight')
	df
	"""
	dataframe[height_col] = dataframe[height_col]/100
	dataframe['bmi'] = dataframe[weight_col] / (dataframe[height_col] * dataframe[height_col])
	dataframe['obesity'] = np.where(dataframe['bmi'] >25 , 1, 0)

	dataframe.drop([height_col, weight_col, 'bmi'], axis=1, inplace=True)

	return dataframe


def generate_countplot(dataframe, data_col, label_col=[]):
	"""Plot a countplot bar graph with labelled proportion. Can further break down proportions into 2 different set of data.

	:param dataframe: dataframe
	:param data_col: name of column containing the data to plot its value counts
	:param label_col: name of column containing labels
	:return: countplot

	>>> generate_countplot(df, 'Survive')
	>>> generate_countplot(df, 'Gender', 'Survive')
	"""
	fig, ax = plt.subplots(figsize=(15, 7))

	# if want to see breakdown by label
	if label_col:
		colours = ['r', 'g']
		totals = dataframe[data_col].value_counts()
		n_hues = dataframe[label_col].unique().size

		ax = sns.countplot(dataframe[data_col],
							hue=dataframe[label_col],
							order=totals.index,
							palette= colours)

		plt.tight_layout()

		temp_totals = totals.values.tolist()*n_hues
		# get count proportion
		for p, t in zip(ax.patches, temp_totals):
			height = p.get_height()
			ax.text(p.get_x()+p.get_width()/2.,
				height + 3,
				'{0:.1%}'.format(height/t),
				ha="center", fontsize=15)

	# plot single data column
	else:
		ax = sns.countplot(x=dataframe[data_col],
			order=dataframe[data_col].value_counts(ascending=False).index)

		abs_values = dataframe[data_col].value_counts(ascending=False)
		rel_values = dataframe[data_col].value_counts(ascending=False, normalize=True).values * 100
		lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

		ax.bar_label(container=ax.containers[0], labels=lbls, fontsize=15)
