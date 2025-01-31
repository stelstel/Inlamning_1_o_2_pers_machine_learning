"""
=================================================
  TEXT CLASSIFICATION WITH MULTI-OUTPUT MODELS
=================================================
This script demonstrates a text classification pipeline using machine learning, with a focus on multi-output classification tasks. The following steps are performed:

Author: Stefan Elmgren
"""

import re
import sys
import warnings
import nltk
import pandas as pd
import numpy as np
import os
import platform
import time
import math

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint

# Save timestamp
start_time = time.time()

# Set to True if more parameters are needed for the evaluation.
# When set to False -> Time consumed = 2 min 3 sec on my computer //Stefan
# When set to True -> Time consumed = 3 min 7 sec on my computer //Stefan
# False -> faster
# True -> slower, better (?)
high_parameter_values = True

################################################################################################################################
def add_results(id, classif, accur, tune_tool = '', best_params= ''):
    """
    Stores the evaluation results of a classifier into a dictionary.

    This function formats and returns a dictionary containing details of a classifier's 
    performance, including its name, accuracy, tuning tool used, and best parameters.

    Parameters:
    -----------
    classif : str
        The name of the classifier being evaluated.
    accur : float or str
        The accuracy score of the classifier, which is converted to a percentage.
    tune_tool : str, optional
        The hyperparameter tuning tool used (e.g., "GridSearchCV" or "RandomizedSearchCV"). 
        Defaults to an empty string if no tuning was applied.
    best_params : str or dict, optional
        The best hyperparameters found during tuning. Defaults to an empty string if tuning was not performed.

    Returns:
    --------
    dict
        A dictionary containing the classifier's name, accuracy (as a percentage), tuning tool, and best parameters.

    Example:
    --------
    >>> add_results("Logistic Regression", 0.92, "GridSearchCV", {"C": 1.0, "penalty": "l2"})
    {'Classifier': 'Logistic Regression', 'Accuracy': 92.0, 'Tuning Tool': 'GridSearchCV', 'Best parameters': {'C': 1.0, 'penalty': 'l2'}}
    """

    accur = float(accur) * 100 # For %
    
    res = {
        "ID" : id,
        "Classifier": classif,
        "Accuracy": accur,
        "Tuning Tool": tune_tool,
        "Best parameters": best_params
    }

    return res
################################################################################################################################



################################################################################################################################
def removeStopWords(sentence):
    """
    Removes stop words from a given sentence.

    Parameters:
    sentence (str): The input sentence from which stop words should be removed.

    Returns:
    str: The sentence with stop words removed.
    
    Dependencies:
    - Requires `nltk.word_tokenize` for tokenization.
    - Assumes `stop_words` is a predefined set or list of stop words.
    
    Example:
    >>> import nltk
    >>> nltk.download('punkt')
    >>> stop_words = {"the", "is", "in", "and"}  # Example stop words
    >>> removeStopWords("The cat is in the garden")
    'cat garden'
    """

    return " ".join(
        [word for word in nltk.word_tokenize(sentence) 
         if word not in stop_words]
    )
################################################################################################################################



nltk.download('punkt_tab')

# Adjust this path to where your NLTK data is located
nltk.data.path.append('C:/Users/sthlm/AppData/Roaming/nltk_data')

# Suppress all warning messages
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# 1. Load the data
data_path = "Book1.csv"
data_raw = pd.read_csv(data_path)

# 2. Shuffle the data
data_raw = data_raw.sample(frac=1)

# 3. Identify category columns
categories = list(data_raw.columns.values)
categories = categories[2:]  # Adjust if your CSV structure changes

# 4. Basic cleaning on the "Heading" column
data_raw['Heading'] = (
    data_raw['Heading']
    .str.lower()
    .str.replace('[^\w\s]', '', regex=True)     # remove punctuation
    .str.replace('\d+', '', regex=True)         # remove digits
    .str.replace('<.*?>', '', regex=True)       # remove HTML tags
)

# 5. Download stopwords and define your stop-word removal function
nltk.download('stopwords')
stop_words = set(stopwords.words('swedish'))

# Clearing the Screen
if platform.system() == "Windows":
    os.system('cls')

data_raw['Heading'] = data_raw['Heading'].apply(removeStopWords)

# 7. Split the data
train, test = train_test_split(data_raw, random_state=42, test_size=0.30, shuffle=True)

train_text = train['Heading']
test_text = test['Heading']

# 8. Vectorize using TF-IDF
vectorizer = TfidfVectorizer(
    strip_accents='unicode', 
    analyzer='word', 
    ngram_range=(1,3), 
    norm='l2')

vectorizer.fit(train_text)

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels=['Id', 'Heading'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels=['Id', 'Heading'], axis=1)


#####################################################
#  PART 1 A: CHOOSE A CLASSIFIER. LogisticRegression
#####################################################
if(high_parameter_values):
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000)) 
else:
    clf = MultiOutputClassifier(LogisticRegression(max_iter=500))
clf.fit(x_train, y_train)
# -------------------------------------------------


########################################################
#  PART 1 B: CHOOSE A CLASSIFIER. RandomForestClassifier
########################################################
if(high_parameter_values):
    estimators = 200
else:
    estimators = 100

clf_rf = RandomForestClassifier(n_estimators=estimators)
# -------------------------------------------------


######################################################
#  PART 1 C: CHOOSE A CLASSIFIER. KNeighborsClassifier
######################################################
if(high_parameter_values):
    clf_knn = KNeighborsClassifier(n_neighbors=9)
else:
    clf_knn = KNeighborsClassifier(n_neighbors=5)

# -------------------------------------------------


#####################################################
#  PART 2: TRAIN YOUR MODEL
#####################################################
clf.fit(x_train, y_train)
clf_rf.fit(x_train, y_train)
clf_knn.fit(x_train, y_train)
# -------------------------------------------------


#####################################################
#  PART 3: MAKE PREDICTIONS
#####################################################
y_pred = clf.predict(x_test)
y_pred_rf = clf_rf.predict(x_test)
y_pred_knn = clf_knn.predict(x_test)
# -------------------------------------------------

results = []

#####################################################
#  PART 4: EVALUATE PERFORMANCE
#####################################################
accuracy = accuracy_score(y_test, y_pred)
results.append(add_results(1, "LogisticRegression", accuracy))

accuracy_rf = accuracy_score(y_test, y_pred_rf)
results.append(add_results(2, "RandomForestClassifier", accuracy_rf))

accuracy_knn = accuracy_score(y_test, y_pred_knn)
results.append(add_results(3, "k-Nearest Neighbors (k-NN)", accuracy_knn))

#####################################################
#  PART 5: TUNE HYPERPARAMETERS
#####################################################

# LogisticRegression, GridSearchCV #################################################################################
param_grid = {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear')  # 'liblinear' supports L1 and L2 penalties
multi_logreg = MultiOutputClassifier(logreg)
grid = GridSearchCV(multi_logreg, param_grid, cv=2, scoring='accuracy') # TODO change cv to 5+
grid.fit(x_train, y_train)
results.append(add_results(4, "LogisticRegression", grid.best_score_, "GridSearchCV", grid.best_params_))


# LogisticRegression, RandomizedSearchCV ##################################################################################################################################################
param_dist = {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear')  # 'liblinear' supports L1 and L2 penalties
multi_logreg = MultiOutputClassifier(logreg)

if(high_parameter_values):
    random_search = RandomizedSearchCV(multi_logreg, param_distributions=param_dist, n_iter=20, cv=10, scoring='accuracy', random_state=42)
else:
        random_search = RandomizedSearchCV(multi_logreg, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

random_search.fit(x_train, y_train)
results.append(add_results(5, "LogisticRegression", random_search.best_score_, "RandomizedSearchCV", random_search.best_params_))


# KNeighborsClassifier, GridSearchCV ############################################################################

# Define parameter grid for KNN
param_grid_knn = {
    'estimator__n_neighbors': [3, 5, 7, 9],  # Number of neighbors to try
    'estimator__weights': ['uniform', 'distance'],  # Weighting strategy
    'estimator__metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
}

# Initialize MultiOutputClassifier with KNeighborsClassifier
knn = KNeighborsClassifier()
multi_knn = MultiOutputClassifier(knn)

# Perform GridSearchCV
if(high_parameter_values):
    grid_knn = GridSearchCV(multi_knn, param_grid_knn, cv=10, scoring='accuracy')
else:
    grid_knn = GridSearchCV(multi_knn, param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(x_train, y_train)

# Store the results
results.append(add_results(6, "KNeighborsClassifier", grid_knn.best_score_, "GridSearchCV", grid_knn.best_params_))


# RandomForestClassifier, RandomizedSearchCV ##########################################################################################################

# Define parameter distributions for RandomizedSearchCV
param_dist_rf = {
    'n_estimators': randint(50, 300),  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of trees
    'min_samples_split': randint(2, 10),  # Minimum number of samples required to split
    'min_samples_leaf': randint(1, 5),  # Minimum number of samples per leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used
}

# Initialize the base RandomForestClassifier
clf_rf_base = RandomForestClassifier(random_state=42)

# RandomizedSearchCV setup
random_search_rf = RandomizedSearchCV(
    clf_rf_base,
    param_distributions=param_dist_rf,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,  # Cross-validation splits (adjust as needed)
    scoring='accuracy',
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the RandomizedSearchCV model
random_search_rf.fit(x_train, y_train)

# Store results
results.append(add_results(
    7,
    "RandomForestClassifier",
    random_search_rf.best_score_,
    "RandomizedSearchCV",
    random_search_rf.best_params_
))


# RandomForestClassifier, GridSearchCV ##################################################################

# Define parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [10, 20],  # Maximum depth
    'min_samples_split': [5, 10],  # Minimum number of samples to split
    'min_samples_leaf': [2, 4],  # Minimum samples per leaf
    'bootstrap': [True, False]  # Whether bootstrap samples are used
}

# Initialize the base RandomForestClassifier
clf_rf_base = RandomForestClassifier(random_state=42)

# GridSearchCV setup
if(high_parameter_values):
    grid_search_rf = GridSearchCV(
        clf_rf_base,
        param_grid=param_grid_rf,
        cv=10,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )
else:
    grid_search_rf = GridSearchCV(
        clf_rf_base,
        param_grid=param_grid_rf,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )

# Fit the GridSearchCV model
grid_search_rf.fit(x_train, y_train)

# Store results
results.append(add_results
    (
        8,
        "RandomForestClassifier",
        grid_search_rf.best_score_,
        "GridSearchCV",
        grid_search_rf.best_params_
    )
)

# -------------------------------------------------


#####################################################
#  PART 6: COMPARE
#####################################################

column_with = 24
table_with = 235
print("Results:")

# Print result headers
for column in results[0]:
    print(f"{column:<{column_with}}", end="\t\t")

print()
print("-" * table_with)

# Sort according to accuracy
results.sort(key=lambda e: e["Accuracy"], reverse=True)  # Descending order


# Print the result rows
for r in results:
    print(f'{r["ID"]:<{column_with}}', end="\t\t")
    print(f'{r["Classifier"]:<{column_with}}', end="\t\t")
    print(f"{r['Accuracy']:.2f} %".ljust(column_with), end="\t\t")
    print(f'{r["Tuning Tool"]:<{column_with}}', end="\t\t")
    print(r["Best parameters"])

print("-" * table_with) 
print()

# Save timestamp
end_time = time.time()

time_seconds = end_time - start_time
minutes = math.floor(time_seconds / 60)  # Get the whole minutes
seconds = int(time_seconds % 60)  # Get the remaining whole seconds

print(f"Time consumed = {minutes} min {seconds} sec")
print()