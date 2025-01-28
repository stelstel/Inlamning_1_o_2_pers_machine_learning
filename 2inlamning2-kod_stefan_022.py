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
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import platform


""" ################################################################################################################################
Stores the results of a classifier's performance into a list.

Parameters:
-----------
res : list
    A list where the classifier's results will be appended. Each entry is a dictionary.
classif : str
    The name of the classifier.
accur : float or str
    The accuracy of the classifier, which will be converted to a percentage.
tune_tool : str, optional
    The tuning tool used for hyperparameter optimization (default is an empty string).
best_params : str or dict, optional
    The best parameters found during tuning (default is an empty string).

Returns:
--------
list
    The updated results list with the new classifier's details.

Example:
--------
>>> results = []
>>> store_results(results, "Random Forest", 0.92, "GridSearchCV", {"n_estimators": 100})
[{'Classifier': 'Random Forest', 'Accuracy': 92.0, 'Tuning Tool': 'GridSearchCV', 'Best parameters': {'n_estimators': 100}}]
""" ################################################################################################################################
def store_results(res, classif, accur, tune_tool = '', best_params= ''):
    accur = float(accur) * 100 # For %
    
    res.append( 
        {
            "Classifier": classif,
            "Accuracy": accur,
            "Tuning Tool": tune_tool,
            "Best parameters": best_params
        }
    )

    return res



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

def removeStopWords(sentence):
    return " ".join(
        [word for word in nltk.word_tokenize(sentence) 
         if word not in stop_words]
    )



# Clearing the Screen
if platform.system() == "Windows":
    os.system('cls')

data_raw['Heading'] = data_raw['Heading'].apply(removeStopWords)

# 6. (Optional) Stemming
# The SnowballStemmer is used for stemming words, meaning it reduces words to their root form (stamform)
# e.g. "bilar" -> 'bil'
# stemmer = SnowballStemmer("swedish")

# def stemming(sentence):
#     stemSentence = ""
#     for word in sentence.split():
#         stemSentence += stemmer.stem(word) + " "
#     return stemSentence.strip()

# If you want to apply stemming, uncomment:
# data_raw['Heading'] = data_raw['Heading'].apply(stemming)

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
#  PART 1: CHOOSE A CLASSIFIER A LogisticRegression
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


#####################################################
#  PART 1: CHOOSE A CLASSIFIER B RandomForestClassifier
#####################################################
estimators = 100 # TODO raise the estimators value //////////////////////////////////////
clf_rf = RandomForestClassifier(n_estimators=estimators)
# -------------------------------------------------


#####################################################
#  PART 1: CHOOSE A CLASSIFIER C KNeighborsClassifier
#####################################################
from sklearn.neighbors import KNeighborsClassifier
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

from sklearn.metrics import accuracy_score, precision_score

accuracy = accuracy_score(y_test, y_pred)
results = store_results(results, "LogisticRegression", accuracy)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
results = store_results(results, "RandomForestClassifier", accuracy_rf)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
results = store_results(results, "k-Nearest Neighbors (k-NN)", accuracy_knn)

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

# GridSearchCV
param_grid = {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear')  # 'liblinear' supports L1 and L2 penalties
multi_logreg = MultiOutputClassifier(logreg)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(multi_logreg, param_grid, cv=2, scoring='accuracy') # TODO change cv to 5+
grid.fit(x_train, y_train)
results = store_results(results, "LogisticRegression", grid.best_score_, "GridSearchCV", grid.best_params_)

# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
param_dist = {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l1', 'l2']}
logreg = LogisticRegression(solver='liblinear')  # 'liblinear' supports L1 and L2 penalties
multi_logreg = MultiOutputClassifier(logreg)
random_search = RandomizedSearchCV(multi_logreg, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42) # TODO change cv to 5+ and n_iter to higher
random_search.fit(x_train, y_train)
results = store_results(results, "LogisticRegression", random_search.best_score_, "RandomizedSearchCV", random_search.best_params_)
# -------------------------------------------------


#####################################################
#  PART 6: COMPARE
#####################################################
"""
If you try multiple classifiers (LogisticRegression, SVC, Random Forest, etc.), 
you can compare their performance. Keep track of the metrics and see which 
classifier + hyperparameter combo works best for this dataset.
"""

# Your CODE HERE:
# -------------------------------------------------
# e.g.,
# classifiers = [LogisticRegression(...), SVC(...), RandomForestClassifier(...)]
# for model in classifiers:
#     ...
#     Evaluate
#     ...
# -------------------------------------------------

column_with = 24

# Print result headers
for column in results[0]:
    print(f"{column:<{column_with}}", end="\t\t")

print()
print("-" * 175)

# Sort according to accuracy
results.sort(key=lambda e: e["Accuracy"], reverse=True)  # Descending order

# Print the result rows
for r in results:
    print(f'{r["Classifier"]:<{column_with}}', end="\t\t")
    print(f"{r['Accuracy']:.2f} %".ljust(column_with), end="\t\t")
    print(f'{r["Tuning Tool"]:<{column_with}}', end="\t\t")
    print(r["Best parameters"])

print("-" * 175)        
print()