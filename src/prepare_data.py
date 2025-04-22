import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils.fixes import percentile
from utils.preprocessing import DataPreprocessor
from IPython.display import display
from joblib import dump
import argparse


class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.preprocessor = DataPreprocessor()

    def load_data(self):
        """Load and initial processing of raw data"""
        self.df = pd.read_csv(self.file_path, index_col=0)
        print(f"\n DataSet Shape: {self.df.shape}")
        return self.df

    def basic_eda(self):
        """Perform exploratory data analysis"""
        print(f"\nDescribe data:\n{self.df.describe()}")
        print(f"\nColumns:\n{self.df.columns}")
        print("\nMissing Values:")
        display(self.df.isna().sum().to_frame('Missing Values').style.background_gradient(cmap='Reds'))
        stats_df = self._statistical_profile(self.df)
        display(stats_df.style.background_gradient(
            cmap='Blues', subset=['Missing %', 'Skewness', 'kurtosis']
        ))
        return stats_df

    def _statistical_profile(self, data):
        """Generate advanced statistical profile"""
        stats_df = pd.DataFrame(index=data.columns)
        stats_df['Data Type'] = data.dtypes
        stats_df['Missing %'] = (data.isnull().mean() * 100).round(2)
        stats_df['Unique Values'] = data.nunique()
        stats_df['Skewness'] = data.skew(numeric_only=True).round(2)
        stats_df['Kurtosis'] = data.kurtosis(numeric_only=True).round(2)

        desc = data.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        stats_df = stats_df.join(desc)
        return stats_df

    def prepare_full_dataset(self):
        """Run complete data preparation pipeline"""
        self.load_data()
        self.basic_eda()
        df_imputed = self.preprocessor.mice_imputation(self.df)
        df_engineered = self.preprocessor.feature_engineering(df_imputed)
        return df_engineered

    def save_prepared_data(self, df, output_path):
        """Save the prepared dataframe as a pickle file"""
        with open(output_path, 'wb') as f:
            dump(df, f)
        print(f"Prepared data saved at: {output_path}")


if __name__ == '__main__':
    # file_path = './data/Titanic Dataset.csv'
    parser = argparse.ArgumentParser(description="Data Preparation Pipeline for Titanic Dataset")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to the save processed pickle file')

    args = parser.parse_args()

    data_prep = DataPreparation(args.input)
    prepared_data = data_prep.prepare_full_dataset()
    data_prep.save_prepared_data(prepared_data, args.output)










































































# from skopt import BayesSearchCV
# import shap
# from google.colab import drive
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, cross_val_score
# from sklearn.metrics import (accuracy_score, precision_score,
#                              recall_score, f1_score, roc_auc_score,
#                              confusion_matrix, classification_report,average_precision_score)
#
# import graphviz
# import dtreeviz
# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer, SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
# from skopt.space import Categorical, Real, Integer
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import phik
# from phik import resources
# drive.mount('/content/drive')
# pd.set_option('display.max_columns',20)
#
#
# plt.style.use("ggplot")
# sns.set_palette('husl')
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['font.family'] = 'DejaVu Sans'
#
#
# # Load dataset
# file_path = '/content/drive/My Drive/ml_data/Titanic Dataset.csv'
# df = pd.read_csv(file_path, index_col=0)
#
# # print(f"\n\033[1mData Dimensions:\033[0m {df.shape}")
# print(f"\n\033[1mDataset Shape: \033[0m\n {df.shape}")
# print(f"\n\033[1mDescribe data\n: \033[0m\n{df.describe()}")
# print(f"\n\033[1mColumns: \033[0m\n {df.columns}")
# print("\n\033[1mMissing Values: \033[0m\n")
# display(df.isna().sum().to_frame('Missing Values').style.background_gradient(cmap='Reds'))
#
#
# def statistical_profile(data):
#   stats_df = pd.DataFrame(index=data.columns)
#   stats_df['Data Type'] = data.dtypes
#   stats_df['Missing %'] = (data.isnull().mean()*100).round(2)
#   stats_df['Unique Values'] = data.nunique()
#   stats_df['Skewness'] = data.skew(numeric_only=True).round(2)
#   stats_df['kurtosis'] = data.kurtosis(numeric_only=True).round(2)
#
#   desc = data.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
#   stats_df = stats_df.join(desc)
#
#   return stats_df
#
# print("\n\033[1mAdvanced Statistical Profile:\033[0m")
# display(statistical_profile(df).style.background_gradient(cmap='Blues', subset=['Missing %', 'Skewness', 'kurtosis']))
#
# # Missing Data imputation using MICE
# def mice_imputation(data, target_var='survived'):
#   numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
#
#   if target_var in numeric_features:
#     numeric_features.remove(target_var)
#
#     # Initialize MICE imputer
#     imputer = IterativeImputer(max_iter=10, random_state=42)
#     data_imputed = data.copy()
#     data_imputed[numeric_features] = imputer.fit_transform(data[numeric_features])
#
#     return data_imputed
#
#
# df_imputed = mice_imputation(df)
# print("\n\033[1mMissing Values After MICE Imputation:\033[0m")
#
# display(df_imputed.isna().sum().to_frame('Missing Values').style.background_gradient(cmap='Greens'))
#
# def feature_engineering(data):
#   df = data.copy()
#
#   # Extract title from name
#   df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#   df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#   df['title'] = df['title'].replace('Mlle', 'Miss')
#   df['title'] = df['title'].replace('Ms', 'Miss')
#   df['title'] = df['title'].replace('Mme', 'Mrs')
#
#   # Family Features
#   df['family_size'] = df['sibsp'] + df['parch'] + 1
#   df['is_alone'] = (df['family_size'] == 1).astype(int)
#
#   # Cabin Features
#   df['cabin_known'] = (~df['cabin'].isna()).astype(int)
#   df['cabin_class'] = df['cabin'].str[0]
#
#   # Fare Features
#   df['fare_per_person'] = df['fare'] / df['family_size']
#   df['fare_category'] = pd.qcut(df['fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
#
#   # Age Features
#   df['age_category'] = pd.cut(df['age'], bins=[0,12,18,35,60,100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
#   return df
#
# df_engineered = feature_engineering(df_imputed)
#
# df_engineered
#
# # df_engineered = advanced_feature_engineering(df_imputed)
#
# def visualizations_advanced(data):
#   # Survival rate by multiple dimensions
#   fig = make_subplots(rows=2, cols=2,
#                       specs=[
#         [{'type': 'xy'}, {'type': 'xy'}],
#         [{'type': 'xy'}, {'type': 'xy'}]
#     ],
#
#                       subplot_titles = ("Survival by Sex and Class",
#                                         "Survival by Age and Fare",
#                                         "Survival by Family Size",
#                                         "Survival by Title"))
#
#   # Plot1: Survival by Sex and Class
#   sex_class_1 = data.groupby(['sex', 'pclass'])['survived'].mean()
#
#   sex_class_df = sex_class_1.reset_index(name='survival_rate')
#   sex_class = sex_class_df.pivot(index='sex', columns='pclass', values='survival_rate')
#
#   # print(sex_class.head())
#   fig.add_trace(go.Bar(x=sex_class.index, y=sex_class[1], name='1st Class'), row=1, col=1)
#   fig.add_trace(go.Bar(x=sex_class.index, y=sex_class[2], name='2nd Class'), row=1, col=1)
#   fig.add_trace(go.Bar(x=sex_class.index, y=sex_class[3], name='3rd Class'), row=1, col=1)
#
#   #Plot2: Survival by Age and Fare
#   fig.add_trace(go.Scatter(x=data['age'], y=data['fare'],
#                            mode='markers',
#                            marker=dict(color=data['survived'],
#                                        colorscale='Viridis',
#                                        showscale=True),
#                            name='Age vs Fare'), row=1, col=2)
#
#
# #Plot 3: Survival by Family Size
#   family_survival = data.groupby('family_size')['survived'].mean()
#   fig.add_trace(go.Bar(x=family_survival.index, y=family_survival.values,
#                        name='Family Size Survival'), row=2, col=1)
#
# # Plot 4: Survival by Title
#   title_survival = data.groupby('title')['survived'].mean().sort_values()
#   fig.add_trace(go.Bar(x=title_survival.index, y=title_survival.values,
#                        name='Title Survival'), row=2,col=2)
#
#   fig.update_layout(height=800, width=1000, title_text="Advanced Survival Analysis", showlegend=False)
#   fig.show()
#
# # Correlation analysis with Phik
#   print("\n\033[1mFeature Correlation Matrix (Phik):\033[0m")
#   cols = ['survived', 'sex', 'age', 'fare', 'family_size', 'title', 'cabin_known']
#   print(data.columns)
#   phik_matrix = data[cols].phik_matrix()
#   plt.figure(figsize=(12,8))
#   sns.heatmap(phik_matrix, annot=True, cmap='coolwarm', center=0)
#   plt.title("Advanced Correlation Analysis (Phik Coefficient)")
#   # fig.update_traces(contours_coloring="fill", contours_showlabels=True)
#   fig.update_traces(marker=dict(color='blue', opacity=0.6), showlegend=True)
#   fig.show()
#
#
# visualizations_advanced(df_engineered)
#
# print(f'\033[1m Last Data Columns : \033[0m{df_engineered.columns}')
#
# def select_features(data):
#   features = [
#       'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
#          'title', 'family_size', 'is_alone', 'cabin_known',
#          'fare_per_person', 'sex_class', 'age_fare',
#          'fare_log', 'age_sqrt', 'title_survival_rate',
#          'class_survival_rate', 'family_status'
#   ]
#   target = 'survived'
#
#   return data[features], data[target]
#
# def create_preprocessor():
#   numerical_features = ['age', 'sibsp', 'parch', 'fare', 'family_size',
#                         'fare_per_person', 'age_fare', 'fare_log', 'age_sqrt',
#                         'title_survival_rate', 'class_survival_rate']
#   categorical_features = ['sex','title', 'sex_class', 'family_status', 'cabin_known']
#
#   numeric_transformer = Pipeline(steps=[
#       ('imputer', SimpleImputer(strategy='median')),
#       ('scaler', StandardScaler())])
#
#   categorical_transformer = Pipeline(steps=[
#       ('imputer', SimpleImputer(strategy='most_frequent')),
#       ('scaler',StandardScaler())])
#
#   preprocessor = ColumnTransformer(
#       transformers=[
#           ('num', numeric_transformer, numerical_features),
#           ('cat', categorical_transformer, categorical_features)])
#
#   return preprocessor
#
# def optimize_hyperparameters(X, y):
#   """Perform Bayesian hyperparameter optimization"""
#   pipeline = Pipeline(steps=[
#       ('preprocessor', create_preprocessor()),
#       ('classifier', DecisionTreeClassifier(random_state=42))])
#
#   search_space = {
#       'classifier__max_depth': Integer(3,15),
#       'classifier__min_samples_split': Integer(2, 20),
#       'classifier__min_samples_leaf': Categorical(['gini', 'entropy']),
#       'classifier__criterion': Categorical(['sqrt', 'log2', None]),
#       'classifier__ccp_alpha': Real(0.0, 0.1, prior='uniform')
#   }
#
#   bayes_cv = BayesSearchCV(
#       estimator=pipeline,
#       search_spaces=search_space,
#       n_iter=50,
#       cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42),
#       n_jobs=-1,
#       scoring='roc_auc',
#       random_state=42)
#
#   bayes_cv.fit(X, y)
#
#   return bayes_cv.best_estimator_, bayes_cv.best_score_
#
# def evaluate_model(model, X, y):
#   """Perform comprehensive model evaluation"""
#
#   # cross-validated metrics
#   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#   cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
#
#   # Train-test split
#   X_train, X_test, y_train, y_test = train_test_split(
#       X, y, test_size=0.2, stratify=y, random_state=42)
#
#   model.fit(X_train, y_train)
#   y_pred = model.predict(X_test)
#   y_proba = model.predict_proba(X_test)[:, 1]
#
#   metrics = {
#       'CV ROC AUC': f"{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}",
#       'Accuracy': accuracy_score(y_test, y_pred),
#       'Precision': precision_score(y_test, y_pred),
#       'Recall': recall_score(y_test, y_pred),
#       'F1 Score': f1_score(y_test, y_pred),
#       'ROC AUC': roc_auc_score(y_test, y_proba),
#       'Average Precision': average_precision_score(y_test, y_proba)
#   }
#
#   # classification report
#   print("\n\033[1mClassification Report:\033[0m")
#   print(classification_report(y_test, y_pred))
#
#   # Confusion matrix
#   plot_confustion_matrix(y_test, y_pred)
#
#   return metrics
#
# def plot_confustion_matrix(y_true, y_pred):
#   """plot annotated confusion matrix"""
#   cm = confusion_matrix(y_true, y_pred)
#   plt.figure(figsize=(8,6))
#   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#               xticklabels=['Perished', 'Survived'],
#               yticklabels=['Perished', 'Survived'])
#   plt.ylabel('Actual')
#   plt.xlabel('Predicted')
#   plt.title('Confusion Matrix')
#   plt.title('Confusion Matrix')
#   plt.show()
#
#   def plot_feature_importance(model, X):
#       """Plot multiple feature importance measures."""
#       preprocessor = model.named_steps['preprocessor']
#       preprocessor.fit(X)
#       X_processed = preprocessor.transform(X)
#       feature_names = X_processed.get_feature_names_out()
#
#       # MDI Importance
#       importances = model.names_steps['classifier'].feature_importances_
#       indices = np.argsort(importances)[::-1]
#
#       plt.figure(figsize=(12, 8))
#       plt.title("Feature Importances (Mean descrease impurity)")
#       plt.barh(range(len(indices)), importances[indices], aligh="center")
#       plt.yticks(range(len(indices)), np.array(feature_names)[indices])
#       plt.gca().invert_yaxis()
#       plt.xlabel("Relative Importance")
#       plt.tight_layout()
#       plt.show()
#
#       # SHAPE Importance
#       explainer = shap.TreeExplainer(model.named_steps['classifier'])
#       shap_values = explainer.shap_values(X_processed)
#
#       plt.figure(figsize=(12, 8))
#       shap.summary_plot(shap_values[1], X_processed, feature_names=feature_names)
#       plt.title("SHAP Value Summary")
#       plt.tight_layout()
#       plt.show()
#
#       def main(file_path):
#           """End-to-end pipeline execution"""
#           # 1. Data Preparation
#           print("Loading and preprocessing data...")
#           df = prepare_data(file_path)
#           X, y = select_features(df)
#
#           # 2. Model Optimization
#           print("\nOptimizing model hyperparameters...")
#           best_model, best_params = optimize_hyperparameters(X, y)
#           print(f"\nOptimal Parameters:\n{best_params}")
#
#           # 3. Model Evaluation
#           print("\nEvaluating model performance...")
#           metrics = evaluate_model(best_model, X, y)
#           print("\nModel Metrics:")
#           for name, value in metrics.items():
#               print(f"{name}: {value}")
#
#           # 4. Visualization
#           print("\nGenerating visualizations...")
#           visualize_tree(best_model, X)
#           plot_feature_importance(best_model, X)
#
#           return best_model