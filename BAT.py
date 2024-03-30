import numpy as np

import seaborn as sns  # statistical data visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from importsall import df,X,y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sns.set_theme(style="whitegrid")





# Set the Bat Algorithm parameters
Qmin = 0
A = 0.5
r = 0.5
Qmax = 2
N = 30
alpha = 0.9
gamma = 0.1
fmin = 0
fmax = 1
max_iter = 100


def bat_algorithm(X_train, X_test, y_train, y_test, A, r, Qmin, Qmax, N, alpha, gamma, fmin, fmax, max_iter):
    # Initialize the population of bats
    f = np.zeros(N)
    v = np.zeros((N, X_train.shape[1]))
    x = np.zeros((N, X_train.shape[1]))
    for i in range(N):
        x[i] = np.random.uniform(0, 1, X_train.shape[1])

    # Find the initial best solution
    best_solution = np.zeros(X_train.shape[1])
    best_fitness = float('-inf')
    for i in range(N):
        if np.random.uniform(0, 1) > r:
            f[i] = fmin + (fmax - fmin) * np.random.uniform(0, 1)
            v[i] = v[i] + (x[i] - best_solution) * f[i]
            x_new = x[i] + v[i]
            x_new = np.clip(x_new, 0, 1)
            if np.random.uniform(0, 1) < A:
                j = np.random.randint(0, N)
                x_new = x_new + alpha * (x[j] - x[i])
            y_pred = train_and_predict(X_train, X_test, y_train, x_new)
            fitness = calculate_fitness(y_test, y_pred)
            if fitness > f[i] and np.random.uniform(0, 1) < gamma:
                x[i] = x_new
                f[i] = fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = x_new

    # Continue the optimization process
    for t in range(1, max_iter):
        for i in range(N):
            f[i] = fmin + (fmax - fmin) * np.random.uniform(0, 1)
            v[i] = v[i] + (x[i] - best_solution) * f[i]
            x_new = x[i] + v[i]
            x_new = np.clip(x_new, 0, 1)
            if np.random.uniform(0, 1) < A:
                j = np.random.randint(0, N)
                x_new = x_new + alpha * (x[j] - x[i])
            y_pred = train_and_predict(X_train, X_test, y_train, x_new)
            fitness = calculate_fitness(y_test, y_pred)
            if fitness > f[i] and np.random.uniform(0, 1) < gamma:
                x[i] = x_new
                f[i] = fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = x_new

            # Select the best features
            selected_features = best_solution.astype(bool)

            # Train a logistic regression model with the selected features
            X_train_selected = X_train.loc[:, selected_features]
            X_test_selected = X_test.loc[:, selected_features]
            model = LogisticRegression(random_state=42)
            model.fit(X_train_selected, y_train)

            # Calculate the accuracy of the model
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)

            return accuracy, selected_features

def train_and_predict(X_train, X_test, y_train, features):
    # Select the features to use for training
    selected_features = features.astype(bool)
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_selected, y_train)

    # Predict the target variable for the test data
    y_pred = model.predict(X_test_selected)

    return y_pred

def calculate_fitness(y_true, y_pred):
# Calculate the accuracy of the predicted target variable
    return accuracy_score(y_true, y_pred)


accuracy, selected_features = bat_algorithm(X_train, X_test, y_train, y_test, A, r, Qmin, Qmax, N, alpha, gamma, fmin, fmax, max_iter)

selected_feature_names = X_train.columns[selected_features]
batattribute = list(selected_feature_names)

#print(batattribute)


from sklearn.preprocessing import MinMaxScaler

# Select the columns you want to normalize
cols_to_normalize = batattribute

# Instantiate a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the selected columns to a normalized scale
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

features = batattribute

# Select the target variable for the feature extraction
target = 'defects'

# Instantiate a SelectKBest object with the chi-squared test
kbest = SelectKBest(chi2, k=2)

# Fit the SelectKBest object to the data and extract the best features
X = df[features]
y = df[target]
X_best = kbest.fit_transform(X, y)

choice = batattribute
from sklearn.preprocessing import MinMaxScaler

# Select the columns you want to normalize
cols_to_normalize = choice

# Instantiate a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the selected columns to a normalized scale
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

features = choice

# Select the target variable for the feature extraction
target = 'defects'

# Instantiate a SelectKBest object with the chi-squared test
kbest = SelectKBest(chi2, k=2)

# Fit the SelectKBest object to the data and extract the best features
X = df[features]
y = df[target]
X_best = kbest.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


ttsbat = [X_train, X_test, y_train, y_test]