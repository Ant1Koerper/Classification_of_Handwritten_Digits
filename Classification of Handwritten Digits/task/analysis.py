import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from helper_functions import fit_predict_eval

(x, y), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

# reshape feature array
x = np.reshape(x, (60000, 28 * 28))

# splint into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x[:6000], y[:6000], test_size=0.3, random_state=40)
# normalize inputs
normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.transform(x_test)

models = {
    "K-nearest neighbours algorithm": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            'n_neighbors': [3, 4],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'brute']
        }
    },
    "Random forest algorithm": {
        "estimator": RandomForestClassifier(random_state=40),
        "param_grid": {
            'n_estimators': [300, 500],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    }
}

# Initialize a dictionary to store results
results = {}

for name, model in models.items():
    best_estimator, accuracy = fit_predict_eval(
        model=model,
        features_train=x_train_norm,
        features_test=x_test_norm,
        target_train=y_train,
        target_test=y_test
    )

    # Store the results
    results[name] = {
        'best_estimator': best_estimator,
        'accuracy': accuracy
    }

# Print the results using the provided template
for name, res in results.items():
    print(f'Model: {name}')
    print(f'best estimator: {res["best_estimator"]}')
    print(f'accuracy: {res["accuracy"]}\n')
