import pandas as pd
import numpy as np
from PIL.ImageOps import expand
from narwhals.selectors import categorical
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:

    def __init__(self):
        self.numerical_features = ['age', 'sibsp', 'parch','fare',
                                   'family_size','fare_per_person','is_alone','has_cabin_info','body','boat']

        self.categorical_features = ['sex', 'title', 'fare_category',
                                     'age_category']

    def mice_imputation(self, data, target_var='survived'):
        """Multivariate imputation by chained equations (MICE)"""
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

        if target_var in numeric_features:
            numeric_features.remove(target_var)

            imputer = IterativeImputer(max_iter=10, random_state=42)
            data_imputed = data.copy()
            data_imputed[numeric_features] = imputer.fit_transform(data[numeric_features])

            return data_imputed

    def feature_engineering(self, data):
        """Create advance features"""
        df = data.copy()

        # Extract title from name
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                           'Jonkheer', 'Dona'], 'Rare')
        df['title'] = df['title'].replace({'Mlle' : 'Miss', 'Ms' : 'Miss','Mme': 'Mrs'})

        # Family Features
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)

        # Cabin Features
        df['has_cabin_info'] = (~df['cabin'].isna()).astype(int)

        # Extract deck letter - core spatial info about the cabin
        df['cabin_class'] = df['cabin'].str[0]

        # Fare features
        df['fare_per_person'] = df['fare'] / df['family_size']
        df['fare_category'] = pd.qcut(df['fare'],
                                      4,
                                      labels=['low', 'Medium', 'High', 'very High'])

        # Age Features
        df['age_category'] = pd.cut(df['age'],
                                    bins=[0, 12, 18, 35, 60, 100],
                                    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

        # Embarked Features (One-hot encode or Label encode)
        df = pd.get_dummies(df, columns=['embarked'])

        # Boat Feature (Binary encoding: 1 if lifeboat assigned ,0 if not)
        df['boat'] = df['boat'].notna().astype(int)

        # Body feature (Binary encoding: 1 if body found, 0 if not)
        df['body'] = df['body'].notna().astype(int)

        # Home Destination Feature (One-hot encode or Label encode)
        df['home.dest'] = df['home.dest'].fillna('Unknown') # Fill missing values with 'Unknown'
        df = pd.get_dummies(df, columns=['home.dest'])

        df.drop(columns=['cabin'], inplace=True)

        return df

    def create_preprocessor(self):

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)])

        return preprocessor
