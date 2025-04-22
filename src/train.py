from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from utils.preprocessing import DataPreprocessor
from utils.models import ModelTrainer
import argparse
import joblib

class TitanicModelTraining:
    def __init__(self,datapreprocessor, modeltrainer):
        self.preprocessor = datapreprocessor
        self.trainer = modeltrainer

    def train_model(self, X, y):
        """Full training pipeline with hyperparameter optimization"""
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor.create_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        search_space = {
            'classifier__max_depth': Integer(3, 15),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__criterion': Categorical(['gini', 'entropy']),
            'classifier__ccp_alpha': Real(0.0, 0.1, prior='uniform')
        }

        best_model = self.trainer.optimize_hyperparameter(
            pipeline, search_space, X, y
        )
        return best_model

    def save_model(self, model, output_path):
        joblib.dump(model, output_path)
        print(f"Model saved at : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Titanic Model training pipline")
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed data (.pkl)')
    parser.add_argument('--model', type=str, required=True, help='Path to save the trained model (.pkl)')
    args = parser.parse_args()

    # Load preprocessed data
    print(f"Loading data from: {args.data}")
    df_engineering = joblib.load(args.data)

    for column in df_engineering.columns:
        print(f"Column : {column}")

    if 'survived'.lower() not in [ col.lower() for col in df_engineering.columns ]:
        raise ValueError("'Survived' column not found in the dataset!")

    X = df_engineering.drop('survived', axis=1)
    y = df_engineering['survived']

    # Instantiate Preprocessor and Trainer
    preprocessor = DataPreprocessor()
    model_optimizer = ModelTrainer()

    # Train model
    trainer = TitanicModelTraining(preprocessor,model_optimizer)
    model = trainer.train_model(X, y)

    # Save trained model
    trainer.save_model(model, args.model)

if __name__ == "__main__":
    main()



















    # from prepare_data import DataPreparation
    #
    # # Prepare datajoblib
    # file_path = './data/Titanic Dataset.csv'
    # DataPrep = DataPreparation(file_path)
    # df_raw = DataPrep.load_data()
    #
    # Preprocessor = DataPreprocessor()
    # # Apply feature Engineering
    # df_engineered = Preprocessor.feature_engineering(df_raw)
    #
    # # df = DataPrep.prepare_full_dataset()
    #
    # # select features
    # X = df_engineered.drop('survived', axis=1)
    # y = df_engineered['survived']
    #
    # # Model - Optimizer
    # modelOptimizer = ModelTrainer()
    #
    # # train model
    # trainer = TitanicModelTraining(Preprocessor,modelOptimizer)
    # model = trainer.train_model(X, y)
    #
    # # Save model
    # import joblib
    # joblib.dump(model, 'titanic_model.pkl')


