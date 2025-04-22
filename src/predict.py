import argparse

import pandas as pd
import joblib
from utils.preprocessing import DataPreprocessor

class TitanicPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model"""
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = DataPreprocessor()
        except Exception as e:
            raise ValueError(f"Model loading failed. {str(e)}")

    def prepare_new_data(self, raw_data):
        """Prepare new data for prediction"""
        try:
            df_imputed = self.preprocessor.mice_imputation(raw_data)
            df_engineered = self.preprocessor.feature_engineering(df_imputed)
            return df_engineered
        except Exception as e:
            raise ValueError(f"Data preparation failed: {str(e)}")

    def predict(self, new_data, return_proba=True):
        """Make Predictions on new data"""
        try:
            processed_data = self.prepare_new_data(new_data)
            predictions = self.model.predict(processed_data)

            results = pd.DataFrame({
                'PassengerId': processed_data.index,
                'Survived': predictions
            })

            if return_proba:
                probabilities = self.model.predict_proba(processed_data)[:, 1]
                results['Survival_Probability'] = probabilities

            return results
        except Exception as e:
            raise RuntimeError(f"Prediction failed. {str(e)}")

def parse_arguments():
    """Configure command-line arguments"""
    parser = argparse.ArgumentParser(description="Titanic Survival Prediction System",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (PKL files)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input csv file.')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Ouput file path for predictions')
    parser.add_argument('--no-proba', action='store_true',
                        help='Exclude probability scores from output')

    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        predictor = TitanicPredictor(args.model)

        # Load and validate input data
        new_data = pd.read_csv(args.input)

        # Generate predictions
        predictions = predictor.predict(new_data, return_proba=not args.no_proba)

        # Save resutls
        predictions.to_csv(args.output, index=False)
        print(f"Success! Predictions saved to {args.output}")
        print(f"Survival rate: {predictions['Survived'].mean():.2%}")

    except Exception as e:
        print(f"Error : {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()