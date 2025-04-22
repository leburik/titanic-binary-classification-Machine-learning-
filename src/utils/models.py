from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

class ModelTrainer:
    def optimize_hyperparameter(self, pipeline, search_space, X, y):
        """Perform Bayesian hyperparameter optimization"""
        bayes_cv = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=50,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1,
            scoring='roc_auc',
            random_state=42)

        bayes_cv.fit(X, y)

        print(f"\nOptimal Parameters:\n{bayes_cv.best_params_}")
        return bayes_cv.best_estimator_