"""
=================================================
  TEXT CLASSIFICATION WITH MULTI-OUTPUT MODELS
=================================================
This script demonstrates a text classification pipeline using machine learning, with a focus on multi-output classification tasks. The following steps are performed:

1. Load data from a CSV file (Book1.csv).
2. Preprocess text data by cleaning, tokenizing, and removing stopwords.
3. Optionally, apply stemming to the text data.
4. Split the dataset into training and testing sets.
5. Transform text data into TF-IDF features.
6. Train multiple machine learning models on the processed text data:
    - Logistic Regression (multi-output classification)
    - Random Forest Classifier
    - k-Nearest Neighbors Classifier
7. Evaluate the models based on accuracy and store the results.
8. Tune hyperparameters using GridSearchCV and RandomizedSearchCV.
9. Compare the performance of the models.

At the end of the script, the following results are generated:
    - Model performance (accuracy)
    - Hyperparameter tuning details (if applicable)
    - Comparison of models based on accuracy and other metrics.

Tasks:
- Implement classifiers and evaluate their performance.
- Optimize models using hyperparameter tuning.
- Compare multiple classifiers to select the best-performing model.
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

################################################################################################################################
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
def add_results(classif, accur, tune_tool = '', best_params= ''):
    accur = float(accur) * 100 # For %
    
    res = {
        "Classifier": classif,
        "Accuracy": accur,
        "Tuning Tool": tune_tool,
        "Best parameters": best_params
    }

    return res
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

################################################################################################################################
def removeStopWords(sentence):
    return " ".join(
        [word for word in nltk.word_tokenize(sentence) 
         if word not in stop_words]
    )
################################################################################################################################



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
"""
In this part, you will:
1. Import or define the classification algorithm(s) you want to try.
2. Examples: LogisticRegression, MultinomialNB (Naive Bayes), SVC, RandomForestClassifier, etc.
3. You can handle multi-label classification if the dataset requires it
   (e.g., OneVsRestClassifier, MultiOutputClassifier, etc.).

Code outline might look like:

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
# or
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

Feel free to experiment with multiple classifiers.
"""

clf = MultiOutputClassifier(LogisticRegression(max_iter=500)) # TODO raise the max_iter value //////////////////////////////////////
clf.fit(x_train, y_train)
# -------------------------------------------------


########################################################
#  PART 1 B: CHOOSE A CLASSIFIER. RandomForestClassifier
########################################################
estimators = 100 # TODO raise the estimators value //////////////////////////////////////
clf_rf = RandomForestClassifier(n_estimators=estimators)
# -------------------------------------------------


######################################################
#  PART 1 C: CHOOSE A CLASSIFIER. KNeighborsClassifier
######################################################
clf_knn = KNeighborsClassifier(n_neighbors=5)
# -------------------------------------------------


#####################################################
#  PART 2: TRAIN YOUR MODEL
#####################################################
"""
Now, train (fit) your chosen classifier on (x_train, y_train).

Example:

clf.fit(x_train, y_train)

After this step, your model will be "trained" on the training data.
"""

clf.fit(x_train, y_train)
clf_rf.fit(x_train, y_train)
clf_knn.fit(x_train, y_train)
# -------------------------------------------------


#####################################################
#  PART 3: MAKE PREDICTIONS
#####################################################
"""
Use your trained model to predict labels on the test set (x_test).
Store the predictions in a variable, e.g. y_pred.
"""

# Your CODE HERE:
# -------------------------------------------------
y_pred = clf.predict(x_test)
y_pred_rf = clf_rf.predict(x_test)
y_pred_knn = clf_knn.predict(x_test)

# -------------------------------------------------

results = []

#####################################################
#  PART 4: EVALUATE PERFORMANCE
#####################################################
"""
Now evaluate how well your model performs using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix, if relevant.

Example with accuracy:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

You can also explore other metrics from sklearn.metrics (e.g., classification_report).
"""

accuracy = accuracy_score(y_test, y_pred)
results.append(add_results("LogisticRegression", accuracy))

accuracy_rf = accuracy_score(y_test, y_pred_rf)
results.append(add_results("RandomForestClassifier", accuracy_rf))


accuracy_knn = accuracy_score(y_test, y_pred_knn)
results.append(add_results("k-Nearest Neighbors (k-NN)", accuracy_knn))


#
# # Possibly more metrics:
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
# -------------------------------------------------


#####################################################
#  PART 5: TUNE HYPERPARAMETERS
#####################################################
"""
Use GridSearchCV or RandomizedSearchCV to find the best parameters
for your chosen classifier.

Example:

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
clf_base = LogisticRegression(solver='liblinear') 
# (solver='liblinear' helps with small data or L1 penalty)

grid = GridSearchCV(clf_base, param_grid, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)

print("Best params: ", grid.best_params_)
print("Best score: ", grid.best_score_)

Then retrain your final model using these best params:

best_clf = grid.best_estimator_
best_clf.fit(x_train, y_train)

Proceed to evaluate as in PART 4.
"""

# LogisticRegression, GridSearchCV #################################################################################
param_grid = {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear')  # 'liblinear' supports L1 and L2 penalties
multi_logreg = MultiOutputClassifier(logreg)
grid = GridSearchCV(multi_logreg, param_grid, cv=2, scoring='accuracy') # TODO change cv to 5+
grid.fit(x_train, y_train)
results.append(add_results("LogisticRegression", grid.best_score_, "GridSearchCV", grid.best_params_))


# LogisticRegression, RandomizedSearchCV ##################################################################################################################################################
param_dist = {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear')  # 'liblinear' supports L1 and L2 penalties
multi_logreg = MultiOutputClassifier(logreg)
random_search = RandomizedSearchCV(multi_logreg, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42) # TODO change cv to 5+ and n_iter to higher
random_search.fit(x_train, y_train)
results.append(add_results("LogisticRegression", random_search.best_score_, "RandomizedSearchCV", random_search.best_params_))


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
grid_knn = GridSearchCV(multi_knn, param_grid_knn, cv=2, scoring='accuracy')  # TODO change cv to 5+
grid_knn.fit(x_train, y_train)

# Store the results
results.append(add_results("KNeighborsClassifier", grid_knn.best_score_, "GridSearchCV", grid_knn.best_params_))


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
    "RandomForestClassifier",
    random_search_rf.best_score_,
    "RandomizedSearchCV",
    random_search_rf.best_params_
))


# RandomForestClassifier, GridSearchCV ##################################################################

# Define parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth
    'min_samples_split': [2, 5, 10],  # Minimum number of samples to split
    'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    'bootstrap': [True, False]  # Whether bootstrap samples are used
}

# Initialize the base RandomForestClassifier
clf_rf_base = RandomForestClassifier(random_state=42)

# GridSearchCV setup
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
results.append(add_results(
    "RandomForestClassifier",
    grid_search_rf.best_score_,
    "GridSearchCV",
    grid_search_rf.best_params_
))

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