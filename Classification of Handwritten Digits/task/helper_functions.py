from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Helps train and evaluate a model using GridSearchCV
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # Extract estimator and parameter grid from the model dictionary
    estimator = model['estimator']
    param_grid = model['param_grid']

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit the grid search to the training data
    grid_search.fit(features_train, target_train)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_

    # Make predictions on the test data using the best estimator
    target_test_predict = best_estimator.predict(features_test)

    # Calculate accuracy
    score = accuracy_score(target_test, target_test_predict)

    return best_estimator, score

