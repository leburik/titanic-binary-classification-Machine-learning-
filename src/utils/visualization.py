import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go


class Visualizer:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.preprocessor = model.named_steps['preprocessor']
        self.X_processed = self.preprocessor.fit_transform(X)
        self.importances = model.named_steps['classifier'].feature_importances_

        plt.style.use("ggplot")
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'DejaVu Sans'

        self.COLORS = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'highlight': '#d62728'
        }

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot annotated confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Perished', 'Survived'],
                    yticklabels=['Perished', 'Survived'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_feature_importance(self, model, X):
        """Plot feature importance using Mean Decrease Impurity (MDI)."""

        # Preprocessor fit and transform
        preprocessor = model.named_steps['preprocessor']
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)

        # Get feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()

        print("Feature names after preprocessing")
        for name in feature_names:
            print(name)

        # Get feature importances from the model
        importances = model.named_steps['classifier'].feature_importances_

        # Sort importances in descending order
        indices = np.argsort(importances)[::-1]

        # Create the plot
        plt.figure(figsize=(14, 9))
        plt.title("Feature Importance (Mean Decrease Impurity)", fontsize=9, weight='bold')

        # Bar plot
        plt.barh(range(len(indices)),
                 importances[indices],
                 align="center",
                 color="seagreen", alpha=0.8)

        # Set y-ticks to feature names and reverse order for better readability
        plt.yticks(range(len(indices)),
                   np.array(feature_names)[indices],
                   fontsize=7)

        # Invert y-axis to show the most important feature at the top
        plt.gca().invert_yaxis()

        # Add labels and grid
        plt.xlabel("Relative Importance", fontsize=7, weight='bold')
        plt.ylabel("Features", fontsize=3, weight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Improve layout for clarity
        plt.tight_layout()

        # Display the plot
        plt.show()

    def advanced_survival_analysis(self, data):
        """Interactive survival analysis visualization"""
        fig = make_subplots(rows=2, cols=2,
                            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                                   [{'type': 'xy'}, {'type': 'xy'}]],

                            subplot_titles=("Survival by Sex and Class",
                                            "Survival by Age and Fare",
                                            "Survival by Family Size",
                                            "Survival by Title"))

        # Plot1 : Survival by Sex and Class
        sex_class_1 = data.groupby(['sex', 'pclass'])['survived'].mean()
        sex_class_df = sex_class_1.reset_index(name='survival_rate')
        sex_class = sex_class_df.pivot(index='sex', columns='pclass', values='survival_rate')

        fig.add_trace(go.Bar(x=sex_class.index, y=sex_class[1], name='1st Class'), row=1, col=1)
        fig.add_trace(go.Bar(x=sex_class.index, y=sex_class[2], name='2nd Class'), row=1, col=1)
        fig.add_trace(go.Bar(x=sex_class.index, y=sex_class[3], name='3rd Class'), row=1, col=1)

        # Plot2: Survival by Age and Fare
        fig.add_trace(go.Scatter(x=data['age'],
                                 y=data['fare'],
                                 marker=dict(color=data['survived'],
                                             colorscale='Viridis',
                                             showscale=True),
                                 name='Age vs Fare'), row=1, col=2)

        # Plot 3: Survival by Family Size
        family_survival = data.groupby('family_size')['survived'].mean()
        fig.add_trace(go.Bar(x=family_survival.index,
                             y=family_survival.values,
                             name='Family Size Survival'),
                      row=2, col=1)

        # Plot 4: Survival by Title
        title_survival = data.groupby('title')['survived'].mean().sort_values()
        fig.add_trace(go.Bar(x=title_survival.index,
                             y=title_survival.values,
                             name='Title Survival'), row=2, col=2)

        fig.update_layout(height=800,
                          width=1000,
                          title_text="Advanced Survival Analysis",
                          showlegend=False)
        fig.show()