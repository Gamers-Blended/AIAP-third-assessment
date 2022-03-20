# AIAP-third-assessment

# Personal Particulars
Full name: Wong Zhi Hang <br>
Email address: wong_zhihang@u.nus.edu


# Overview of Folder and Folder Structure
```
├── src
│ └── main.py
| └── config.yaml
├── README.md
├── eda.ipynb
├── requirements.txt
└── run.sh
```
**Src**
The `src` folder contains 2 files: `main.py` and `config.yaml`.
`main.py` 
- This is the Python script that performs the entire end-to-end machine learning pipeline (collect data from `survive.db`, process it, fit it into a Machine Learning model, evaluate its result).

`config.yaml` 
- This file contains the current configuration for the pipeline. <br> Open this file to change the algorithms and their parameters.
- Users can also set their target variables, decide which variables to remove and normalise.
- Users can set the test size of the data.

**`README.md`**
- This is the Read Me file that explains the pipeline design and its usage.

**`eda.ipynb`**
- This is the notebook that presents the findings of my analysis on the `survive.db` dataset.
- It outlines the steps taken in Data Cleaning, Feature Engineering and Data Exploration, as well as the thought process behind each step.
- It identifies the variables that are useful for predicting survivial rate.
- It also contains insights and observations that are visualised with charts.

**`requirements.txt`**
- This text file contains the dependencies needed to be installed for the `run.sh` and `main.py` to function.

**`run.sh`**
- Run this shell file to run `main.py`, and install any dependencies listed in `requirements.txt`.


# Instructions
## Executing the Pipeline
Note: This was done on a Windows 10 system with WSL installed. <br>
1. Ensure that the every file is correctly placed accordingly to the folder structure outlined in **Overview of Folder and Folder Structure**.
2. [WSL] Place the base folder in `home/<username>/`
3. [WSL] Open the Ubuntu terminal.
4. Ensure that the current directory is set at the root of this base folder.
5. Run this command: `bash run.sh`
6. The results will be displayed on the terminal.

## Modifying Parameters
1. Open the `config.yaml` file located in the `src` folder.
2. Under `# INITIAL SETTINGS`, you can change the target variable, test size, columns to remove and normalise. A brief description is provided as comments for each of these settings. <br> **There is a list of columns that must not be removed (stated in the comment), else the program will give an error.**
3. Under `# model selector`, choose any 1 of the 3 algorithms provided: k-nearest Neighbors (`kNN`), Random Forest (`Rf`) and Support Vector Classifier (`SVC`). <br> Type in either 1 of these symbols to select them: `kNN`, `Rf`, `SVC`
4. Based on the model selected, you can adjust its parameters accordingly. <br> Check the first comment of each block after `# model selector` to find the set of parameters for the corresponding model. <br> The acceptable inputs are describled in the attached comments for each parameter.
5. Save any changes made.
6. Run `run.sh` to see the results based on the changes made.


# Steps of the Pipeline
1. Import any libraries and dependencies needed to run the tasks stated in `main.py`.
2. Load the configuration settings stated in `config.yaml` into `main.py`.
3. Establish a connection with the `survive.db` database file using `sqlite3`. The retrieved data will be exported into a `pandas` DataFrame. Close the connection once the data has been retrieved.
4. Remove entries with missing data.
5. For a consistent naming convention, replace all whitespaces in every column name with an underscore.
6. Remove any redundant columns as stated in `config.yaml`.
7. Manually perform One-Hot Encoding for all catergoical variables.
8. Convert all negative values in `Age` to positive values.
9. Determine whether a patient has `obesity` from their `Height` and `Weight`.
10. Remove the `Height` and `Weight` columns and the intermediate `bmi` column since they are not needed for the model.
11. Remove any suspected synthetic data.
12. Normalise all numerical variables.
13. Create independent and depedendent variables.
14. Split the data into train and test set.
15. Train the selected model with its corresponding parameters as stated in `config.yaml`.
16. Evaluate the model's predicted results.


# Key Findings from EDA (Task 1)
- The dataset is imbalanced as the number of surviving and non-surviving patients are not close. <br> This would be addressed in the algorithms and metrics used.
- `Creatine_phosphokinase` is not useful in predicting survival rate. Hence it is removed during processing.
- `Height` and `Weight` have shown to have high multicollinearity, hence both can be combined to obtain Body Mass Index, which was used for the `obesity` indicator variable in the Feature Engineering step.
- Further details on how these conclusions were established can be found in `eda.ipynb`.


# Explanation of Model Choices
The following models were used:
1. k-nearest Neighbors Classifier
2. Random Forest Classifier
3. Support Vector Classifier 

## k-nearest Neighbors Classifier
It is non-parametric as it does not make any assumptions on the data fed into it (e.g. linearity, conditional independence). Hence it is useful when using real-world data since some of the variables are dependent on one another. <br>
Furthermore, it trains very quickly as it does not derive any discriminative function from the training data. <br>

## Random Forest Classifier
As it based on the bagging algorithm and uses Ensemble Learning technique, it reduces the overfitting problem in Decision Trees. <br>
Although missing values were removed and all numerical values have been normalised in the processing step, these steps are not required when using Random Forest. It automatically handles missing values and uses a rule based approach instead of distance calculation. <br>
Morevoer, it would not be affected by the high non-linearity between independent variables. <br>
It also can be configurated to account for imbalanced datasets with `class_weight`.

## Support Vector Classifier
It is able to handle high dimensional data. <br>
The kernel allows inputs to be converted into high dimensional data, thus eliminating the assumption that data is linearly separable. <br>
It also can be configurated to account for imbalanced datasets with `class_weight`.

# Evaluation of Models
The following metrics were used:
1. F2 Score
2. Kappa Score
3. Confusion Matrix

Accuracy can be misleading for an imbalanced dataset. Since there are more non-surviving patients than surviving ones, the model could still achive a high accuracy score just from predicting that every patient will not survive. Hence precision-based metrics would give a more truthful idea of the model's performance.

## F2 Score
```
               | Positive Prediction | Negative Prediction
Positive Class | True Positive       | False Negative 
Negative Class | False Positive      | True Negative 
```
Precision =  True Positives / (True Positives + False Positives) <br>
Recall = True Positives / (True Positives + False Negatives) <br>
F2 Score = (5 * Precision * Recall) / (4 * Precision + Recall) <br>
In the context of predicting survival rates, False Negatives (unable to predict the patient will die) are more important. The F2 Score lowers the importance of precision and increases the importance of recall. <br>
Maximising recall minimizes false negatives. Thus F2 Score focuses on minimising false negatives than minimizing false positives.

## Kappa Score
It reflects prediction performance of smaller classes (surviving patients) since accuracy is normalised by the imbalance of the classes in the dataset. <br>
Moreover, it also removes the possibility of the classifier and a random guess agreeing and measures the number of predictions it makes that cannot be explained by a random guess.

## Confusion Matrix
The Confusion Matrix displays both the number of correct and incorrect predictions for each class. It gives a good view of how the model is exactly performing for each particular class.

# Other Considerations
- Could try ensembling different models (besides Random Forest) for a better accuracy.
- Another way to tackle imbalanced datasets is to generate synthetic samples by randomly sampling the attributes from the minority class (surviving patients). Synthetic Minority Over-sampling Technique (SMOTE) is a common method used to oversample the minority class.

# Update
Technical improvements were made upon clearing the Software Engineering track. <br>
Original code is commented out.

## Improvement 1
In `eda.ipynb`, at the very start of the notebook, all custom functions were placed into a single package named `utils.py`. The package in stored in the `src` folder. `main.py` also imports these functions from this file.

## Improvement 2
In `eda.ipynb`, under **Data Cleaning**, the loop that prints out all unique values of each column of the dataframe has been replaced with a `lambda` function.

## Improvement 3
In `eda.ipynb`, under **Check for Outliers**, the loop that calls the `iqr_outliers` function on each numerical column has been replaced with the `apply` function.

## Improvement 4
In `utils.py`, docstrings were added to explain the functionalities of the custom functions in `utils.py`, and how to use them.

## Improvement 5
In `main.py`, the system will catch Value Type errors and exceptions before spliting the data into train and test sets.
