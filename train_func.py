##### Grid Search To find suitable parameters for Random Forest Classifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train(X_train, y_train, X_test, y_test, param_grid=None, cv=5):

        # Define parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400],      # number of trees
            'max_depth': [None, 10, 20, 30, 40],     # depth of trees
            'min_samples_split': [2, 5, 10, 20],     # min samples to split a node
            'min_samples_leaf': [1, 2, 4, 8],       # min samples per leaf
            'max_features': ['sqrt', 'log2']     # number of features to consider
        }

    model = RandomForestClassifier(random_state=42)

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,                # 5-fold cross-validation
        n_jobs=-1,           # use all CPU cores
        verbose=2,
        scoring='f1_macro'   # f1_macro balances across classes
    )
        
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    return best_model, report





