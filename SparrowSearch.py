import numpy as np
from math import exp
from sklearn.model_selection import train_test_split

from importsall import X, y, df


def SSA(X_train, y_train, X_test, y_test, n, max_iter, alpha, beta):
    def objective_function(position):
        selected_features = [col for col, value in position.items() if value == 1]
        # Here you can write your own code to evaluate the selected features
        # For example, you can use a classification algorithm to evaluate the performance of the selected features
        # and return a fitness score, or use any other evaluation metric that suits your problem
        # In this example, we assume that the objective function is the sum of the selected features
        return np.sum([value for col, value in position.items() if value == 1])

    positions = [{col: np.random.randint(0, 2) for col in X_train.columns} for _ in range(n)]
    fitness = np.zeros(n)
    for i in range(n):
        fitness[i] = objective_function(positions[i])

    best_position = positions[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    for t in range(max_iter):
        distances = np.zeros(n)
        directions = np.zeros((n, len(X_train.columns)))
        for i in range(n):
            distances[i] = exp(-beta * fitness[i] / (best_fitness + 1e-100))
            if distances[i] > 1e-10:
                selected_features = [col for col, value in positions[i].items() if value == 1]
                j = np.random.choice([k for k in range(n) if k != i])
                selected_features_j = [col for col, value in positions[j].items() if value == 1]
                for k, col in enumerate(X_train.columns):
                    if col in selected_features and col in selected_features_j:
                        directions[i][k] += alpha * distances[i] * (positions[j][col] - positions[i][col])
                    elif col in selected_features and col not in selected_features_j:
                        directions[i][k] += alpha * distances[i] * (best_position[col] - positions[i][col])
                    elif col not in selected_features and col in selected_features_j:
                        directions[i][k] += alpha * distances[i] * (positions[j][col] - best_position[col])
                    else:
                        directions[i][k] += 0
            directions[i] = directions[i] / (np.linalg.norm(directions[i]) + 1e-100)

        for i in range(n):
            new_position = {}
            for j, col in enumerate(X_train.columns):
                new_position[col] = positions[i][col] + directions[i][j] * np.random.normal(0, 1)
                new_position[col] = np.clip(new_position[col], 0, 1)
            fitness[i] = objective_function(new_position)
            positions[i] = new_position

        if np.max(fitness) > best_fitness:
            best_position = positions[np.argmax(fitness)]
            best_fitness = np.max(fitness)

    return best_position


# Load the CSV file

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set the parameters for the Sparrow Search Algorithm
n = 20  # number of sparrows
max_iter = 100  # maximum number of iterations
alpha = 0.5  # step size parameter
beta = 1  # influence parameter for fitness

# Print features before selection
L = X_train.columns.tolist()
print("Features before Sparrow Search Algorithm: ", len(L))
print(L)  # Assuming X_train represents features before selection

# Run the Sparrow Search Algorithm
selected_features = SSA(X_train, y_train, X_test, y_test, n, max_iter, alpha, beta)

# Print the selected
Listt = []
# print(f'Selected Features:{selected_features.items()}')
for col, value in selected_features.items():
    if value == 1:
        Listt.append(str(col))

SSA = Listt

from sklearn.preprocessing import MinMaxScaler

# Select the columns you want to normalize
cols_to_normalize = SSA

# Instantiate a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the selected columns to a normalized scale
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Print selected features after SSA
selected_cols = [col for col, value in selected_features.items() if value == 1]
print("Features selected by Sparrow Search Algorithm: ", len(selected_cols))
print(selected_cols)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

features = SSA

# Select the target variable for the feature extraction
target = 'defects'

# Instantiate a SelectKBest object with the chi-squared test
kbest = SelectKBest(chi2, k=2)

# Fit the SelectKBest object to the data and extract the best features
X = df[features]
y = df[target]
X_best = kbest.fit_transform(X, y)

choice = SSA
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

ttsssa = [X_train, X_test, y_train, y_test]
