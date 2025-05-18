# %% [markdown]
# # Submission Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

# %% [markdown]
# - Nama: Bayu Indra Kusuma
# - Email: bayuindrakusuma05@gmail.com
# - Id Dicoding: bayuik

# %% [markdown]
# ## Persiapan

# %% [markdown]
# ### Menyiapkan library yang dibutuhkan

# %%
# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import math
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from imblearn.over_sampling import SMOTE
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# %% [markdown]
# ### Menyiapkan data yang akan diguankan

# %%
df = pd.read_csv('data.csv', sep=';')
df.head()

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# #### Helper Functions
#
# This section defines reusable functions that will be used in the following steps of the project, such as preprocessing, visualization, or evaluation.

# %%
def check_data_quality(dataframe, verbose=True):
    """
    Analyze dataset quality including basic info, missing values, and duplicates.

    Parameters:
        dataframe (pd.DataFrame): Input dataframe to analyze
        verbose (bool): If True, prints detailed analysis output

    Returns:
        dict: Quality report containing:
            - data_types: DataFrame dtypes
            - null_counts: Series of null counts per column
            - duplicate_count: Number of duplicate rows
            - total_rows: Total rows in dataframe
            - total_columns: Total columns in dataframe
    """

    # Initialize quality report dictionary to store results
    quality_report = {
        'data_types': None,
        'null_counts': None,
        'duplicate_count': 0,
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns)
    }

    # 1. Check data types and basic info
    if verbose:
        print("\n=== DATA TYPE INFORMATION ===")
        print(dataframe.info())

    # Store data types in report
    quality_report['data_types'] = dataframe.dtypes

    # 2. Analyze missing values
    null_counts = dataframe.isnull().sum()
    quality_report['null_counts'] = null_counts

    if verbose:
        print("\n=== MISSING VALUE ANALYSIS ===")
        print(null_counts)

        # Show percentage if there are missing values
        if null_counts.sum() > 0:
            print("\nMissing Value Percentages:")
            print((null_counts / len(dataframe)) * 100)

    # 3. Check for duplicate rows
    duplicate_count = dataframe.duplicated().sum()
    quality_report['duplicate_count'] = duplicate_count

    if verbose:
        print("\n=== DUPLICATE DATA ANALYSIS ===")
        print(f"Duplicate row count: {duplicate_count}")
        print(f"Duplicate percentage: {(duplicate_count / len(dataframe)) * 100:.2f}%")

    return quality_report

# %%
def analyze_numerical_data(dataframe, specific_cols=None):
    """
    Analyze numeric data with correlation and distribution visualization.

    Parameters:
    - dataframe: DataFrame to be analyzed
    - specific_cols: List of specific columns (optional). If None, will retrieve all numeric columns
    """
    # Select numeric columns
    numeric_cols = specific_cols if specific_cols else \
        dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Validate if there are no numeric columns
    if not numeric_cols:
        print("No numeric columns found in the dataset.")
        return

    # 1. Correlation Analysis
    print("\n=== Correlation Analysis ===")
    correlation_data = dataframe[numeric_cols].corr()

    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(
        correlation_data,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        center=0
    )
    heatmap.set_title('Correlation Matrix of Numerical Features', pad=20, fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 2. Distribution Analysis with Boxplots (3 per row)
    print("\n=== Distribution Analysis ===")
    num_features = len(numeric_cols)
    cols = 3  # Number of plots per row
    rows = math.ceil(num_features / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_cols):
        ax = axes[i]
        sns.boxplot(data=dataframe, y=feature, ax=ax, color='#72A0C1', width=0.4)

        # Calculate statistics
        median = dataframe[feature].median()
        q1 = dataframe[feature].quantile(0.25)
        q3 = dataframe[feature].quantile(0.75)

        # Add annotation
        ax.text(0.05, 0.95,
                f"Median: {median:.2f}\nQ1: {q1:.2f}\nQ3: {q3:.2f}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=8)

        ax.set_title(f'Distribution of {feature}', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel(feature, fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# %%
def plot_boxplots_by_status(df, numerical_cols, target_col='Status', plots_per_row=2):
    """
    Plot boxplots for each numerical column grouped by a categorical target column.

    Parameters:
    - df: DataFrame containing the data
    - numerical_cols: List of numerical column names to plot
    - target_col: Categorical column to group by (default: 'Status')
    - plots_per_row: Number of plots per row (default: 2)
    """
    total = len(numerical_cols)
    rows = math.ceil(total / plots_per_row)

    fig, axes = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 6, rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.boxplot(data=df, x=target_col, y=col,hue=target_col, palette='coolwarm', ax=axes[i])
        axes[i].set_title(f'Boxplot of {col} by {target_col}', fontsize=13)
        axes[i].set_xlabel(target_col, fontsize=11)
        axes[i].set_ylabel(col, fontsize=11)
        axes[i].tick_params(axis='x', rotation=0, labelsize=10)

    # Remove any extra empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# %%
def plot_status_distribution_pie(df, status_col='Status'):
    """
    Plot a pie chart showing the distribution of a categorical status column.

    Parameters:
    - df: DataFrame containing the data
    - status_col: Name of the categorical column to analyze (default: 'Status')
    """
    status_counts = df[status_col].value_counts()

    plt.figure(figsize=(5, 5))  # Compact size
    plt.pie(
        status_counts,
        labels=status_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("Set2")[:len(status_counts)],  # Changed palette
        explode=[0.05] * len(status_counts),
        wedgeprops={'edgecolor': 'white'}
    )

    plt.title(f'Distribution of {status_col}', fontsize=14)
    plt.tight_layout()
    plt.show()


# %%
def plot_numerical_boxplots(df, numerical_cols, plots_per_row=2):
    """
    Display boxplots for each numerical feature in a grid layout.

    Parameters:
    - df: DataFrame containing the data
    - numerical_cols: List of numerical column names to plot
    - plots_per_row: Number of plots per row (default: 2)
    """
    total = len(numerical_cols)
    rows = math.ceil(total / plots_per_row)

    fig, axes = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 6, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.boxplot(data=df, y=col, ax=axes[i], color='#FFA07A', width=0.4)  # Soft salmon color
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(col, fontsize=10)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_categorical_countplots(df, category_cols, target_col='Status'):
    """
    Display countplots for each categorical feature against a target column.

    Parameters:
    - df: DataFrame containing the data
    - category_cols: List of categorical column names to plot
    - target_col: Column to use for hue (default: 'Status')
    """
    for col in category_cols:
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(data=df, y=col, hue=target_col, palette='Set2', dodge=True)

        # Add count labels
        for container in ax.containers:
            ax.bar_label(container, fontsize=9, padding=2, label_type='edge')

        plt.title(f'{col} vs {target_col}', fontsize=12)
        plt.xlabel('Count')
        plt.ylabel('')
        plt.legend(title=target_col, fontsize=9, loc='upper right')
        plt.tight_layout()
        plt.show()


# %%
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        for col in self.cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - 1.5 * IQR
            self.upper_bounds[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.cols:
            X[col] = np.clip(X[col], self.lower_bounds[col], self.upper_bounds[col])
        return X

# %% [markdown]
# #### Data Information

# %%
check_data_quality(df)

# %% [markdown]
# ## Data Preparation / Preprocessing

# %%
# Constant mappings for qualifications and occupations
QUALIFICATION_MAP = {
    1: 'Secondary Education - 12th Year of Schooling or Eq.',
    2: "Higher Education - Bachelor's Degree",
    3: 'Higher Education - Degree',
    4: "Higher Education - Master's",
    5: 'Higher Education - Doctorate',
    6: 'Frequency of Higher Education',
    9: '12th Year of Schooling - Not Completed',
    10: '11th Year of Schooling - Not Completed',
    11: '7th Year (Old)',
    12: 'Other - 11th Year of Schooling',
    14: '10th Year of Schooling',
    18: 'General commerce course',
    19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
    22: 'Technical-professional course',
    26: '7th year of schooling',
    27: '2nd cycle of the general high school course',
    29: '9th Year of Schooling - Not Completed',
    30: '8th year of schooling',
    34: 'Unknown',
    35: "Can't read or write",
    36: 'Can read without having a 4th year of schooling',
    37: 'Basic education 1st cycle (4th/5th year) or equiv.',
    38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
    39: 'Technological specialization course',
    40: 'Higher education - degree (1st cycle)',
    41: 'Specialized higher studies course',
    42: 'Professional higher technical course',
    43: 'Higher Education - Master (2nd cycle)',
    44: 'Higher Education - Doctorate (3rd cycle)'
}

OCCUPATION_MAP = {
    0: 'Student',
    1: 'Legislative/Executive Bodies, Directors and Managers',
    2: 'Intellectual and Scientific Specialists',
    3: 'Intermediate Technicians and Professionals',
    4: 'Administrative Staff',
    5: 'Services, Security, Sellers',
    6: 'Skilled Agriculture/Fisheries/Forestry Workers',
    7: 'Skilled Industrial/Construction Workers',
    8: 'Machine Operators and Assemblers',
    9: 'Unskilled Workers',
    10: 'Armed Forces',
    90: 'Other Situation',
    99: 'Blank',
    122: 'Health Professionals',
    123: 'Teachers',
    125: 'ICT Specialists',
    131: 'Science and Engineering Technicians',
    132: 'Health Technicians',
    134: 'Legal/Social/Sports/Cultural Technicians',
    141: 'Office Workers and Secretaries',
    143: 'Accounting/Financial Operators',
    144: 'Other Admin Support Staff',
    151: 'Personal Service Workers',
    152: 'Sellers',
    153: 'Personal Care Workers',
    171: 'Construction Workers (Non-electricians)',
    173: 'Precision Workers, Artisans',
    175: 'Food/Wood/Clothing Industry Workers',
    191: 'Cleaners',
    192: 'Unskilled Agricultural Workers',
    193: 'Unskilled Industrial/Transport Workers',
    194: 'Meal Preparation Assistants'
}

# Mapping configuration for categorical columns
COLUMN_MAPPINGS = {
    'Marital_status': {
        1: 'single', 2: 'married', 3: 'widower',
        4: 'divorced', 5: 'facto union', 6: 'legally separated'
    },
    'Application_mode': {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student',
        16: 'Special contingent (Madeira)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Diff Plan - 533-A/99 b2',
        27: 'Other Inst - 533-A/99 b3',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Course change',
        44: 'Tech diploma holders',
        51: 'Institution/course change',
        53: 'Short cycle diploma',
        57: 'Institution/course change (International)'
    },
    'Course': {
        33: 'Biofuel Production',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (Evening)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Marketing Management',
        9773: 'Journalism',
        9853: 'Basic Education',
        9991: 'Management (Evening)'
    },
    'Daytime_evening_attendance': {0: 'evening', 1: 'daytime'},
    'Previous_qualification': {
        1: 'Secondary education',
        2: "Bachelor's degree",
        3: 'Higher education - degree',
        4: "Master's degree",
        5: 'Doctorate',
        6: 'Ongoing higher education',
        9: '12th grade - not completed',
        10: '11th grade - not completed',
        12: 'Other - 11th grade',
        14: '10th grade',
        15: '10th grade - not completed',
        19: 'Basic education - 3rd cycle',
        38: 'Basic education - 2nd cycle',
        39: 'Technological course',
        40: 'Higher education (1st cycle)',
        42: 'Professional tech course',
        43: 'Master’s (2nd cycle)'
    },
    'Nacionality': {
        1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian',
        13: 'Dutch', 14: 'English', 17: 'Lithuanian', 21: 'Angolan',
        22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican', 26: 'Santomean',
        32: 'Turkish', 41: 'Brazilian', 62: 'Romanian', 100: 'Moldovan',
        101: 'Mexican', 103: 'Ukrainian', 105: 'Russian',
        108: 'Cuban', 109: 'Colombian'
    },
    'Mothers_qualification': deepcopy(QUALIFICATION_MAP),
    'Fathers_qualification': deepcopy(QUALIFICATION_MAP),
    'Mothers_occupation': deepcopy(OCCUPATION_MAP),
    'Fathers_occupation': deepcopy(OCCUPATION_MAP),
    'Displaced': {0: 'no', 1: 'yes'},
    'Educational_special_needs': {0: 'no', 1: 'yes'},
    'Debtor': {0: 'no', 1: 'yes'},
    'Tuition_fees_up_to_date': {0: 'no', 1: 'yes'},
    'Gender': {0: 'female', 1: 'male'},
    'Scholarship_holder': {0: 'no', 1: 'yes'},
    'International': {0: 'no', 1: 'yes'}
}

# Apply mappings to the corresponding columns if they exist in the DataFrame
for col_name, mapping in COLUMN_MAPPINGS.items():
    if col_name in df.columns:
        df[col_name] = df[col_name].map(mapping)


# %%
check_data_quality(df)

# %%
# Define categorical columns
categorical_columns = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
    'Nacionality', 'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation',
    'Fathers_occupation', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International', 'Status'
]

# Cast selected columns to categorical dtype
df[categorical_columns] = df[categorical_columns].apply(pd.Categorical)

# Show dataframe summary
check_data_quality(df)

# %%
# This cell provides a statistical summary of all categorical features in the dataset using `.describe()`. It helps understand the distribution, unique values, and frequency of categories.
df.describe(include='category').T

# %%
df.describe(exclude='category').T

# %%
# Calculate number of columns
num_columns = len(df.columns)
print(f"Number of columns: {num_columns}")

# %%
# The cleaned and preprocessed dataset is saved to a CSV file for future use or for deployment purposes.
df.to_csv('preprocessed_school_data.csv', index=False)

# %%
# This step uploads the preprocessed dataset to a PostgreSQL database using SQLAlchemy, enabling centralized storage and access from other applications or services.
connection_url = "postgresql://postgres.schsfsjdxviyhopifisp:gLqKAqZpWgN7pgjm@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

engine = create_engine(connection_url)
table_name = 'preprocessed_school_data'

df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

# %%
# List of features to analyze (customizable)
FEATURES_TO_ANALYZE = [
    'Admission_grade',
    'Age_at_enrollment',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate',
    'Inflation_rate',
    'GDP'
]

analyze_numerical_data(df, FEATURES_TO_ANALYZE)

# %%
# In this section, we define:
# - Columns to drop from the dataset.
# - Numerical columns to be processed.
# - Categorical columns for encoding or transformation.

# These lists help streamline preprocessing steps later in the pipeline.


columns_to_drop = [
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_without_evaluations'
]

columns_to_process = [
    'Admission_grade', 'Age_at_enrollment',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
]

numerical_cols = ['Admission_grade', 'Age_at_enrollment',
                  'Curricular_units_1st_sem_enrolled',
                  'Curricular_units_1st_sem_evaluations',
                  'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
                  'Curricular_units_2nd_sem_enrolled',
                  'Curricular_units_2nd_sem_evaluations',
                  'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
                  'Unemployment_rate', 'Inflation_rate', 'GDP']

category_cols = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International'
]


# %%
# This section removes unnecessary columns and applies an outlier handling pipeline to selected numerical features. The fitted pipeline is saved using `joblib` for reuse during inference.
df = df.drop(columns=columns_to_drop)

outlier_pipeline = Pipeline([
    ('outlier_handler', OutlierHandler(cols=columns_to_process))
])

df = outlier_pipeline.fit_transform(df)

joblib.dump(outlier_pipeline, 'model/outlier_preprocessor_dropout.joblib')

# %%
# Boxplots are used to inspect the distribution and detect outliers in numerical features, while countplots show the frequency distribution of categorical features.
plot_numerical_boxplots(df, numerical_cols)
plot_categorical_countplots(df, category_cols)

# %%
check_data_quality(df)

# %%
plot_boxplots_by_status(df, numerical_cols)

# %%
plot_status_distribution_pie(df)

# %%
# Clean up whitespace from values in the 'Status' column
df['Status'] = df['Status'].astype(str).str.strip()

# Remove rows where 'Status' is 'Enrolled' and reset the index
df = df[df['Status'] != 'Enrolled'].reset_index(drop=True)

df['Status'].value_counts()

# %%
df.head()

# %%
check_data_quality(df)

# %%
# This cell displays rows where the 'Fathers_qualification' column has missing values, helping identify patterns or decide how to handle them.
df[df['Fathers_qualification'].isnull()]

# %%
# All rows with missing values are removed from the dataset to ensure clean input for modeling.
df.dropna(inplace=True)

# %%
check_data_quality(df)

# %%
# Identify columns with categorical data
category_columns = df.select_dtypes(include=['category', 'object']).columns

# Dictionary to store label encoders for each categorical column
encoders = {}

# Apply label encoding to each categorical column
for column in category_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder  # Store the encoder for future use (e.g., inverse transform)

# Ensure the directory exists before saving
os.makedirs("model", exist_ok=True)

# Save the encoders to a file using joblib
joblib_path = "model/encoder.joblib"
joblib.dump(encoders, joblib_path)

print(f"Encoders successfully saved to: {joblib_path}")

# %%
df.head()

# %%
# The final processed dataset is saved as a CSV file for further use in modeling or analysis.
df.to_csv('student_status_processed.csv', index=False)

# %% [markdown]
# ## Modeling

# %%
# The dataset is split into features (`X`) and target variable (`y`), where `Status` is the target to be predicted.
X = df.drop(columns=['Status'])
y = df['Status']

# %%
# The dataset is split into training and testing sets using stratified sampling to maintain the distribution of the target variable. 80% of the data is used for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# This cell shows the class distribution in the original (unbalanced) training set before applying SMOTE.
y_train.value_counts()

# %%
# SMOTE is applied to the training data to balance the class distribution by generating synthetic samples for the minority classes. The new class distribution is then displayed.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

y_resampled.value_counts()

# %%
# Displays the list of feature column names after applying SMOTE to ensure all expected features are present.
X_resampled.columns

# %%
# Displays the resampled feature dataset after SMOTE has been applied to the training data.
X_resampled

# %%
# A pipeline is created to handle preprocessing and modeling. Numerical features are scaled, categorical features are encoded, and a Random Forest classifier is added as the final step.
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, category_cols)
    ]
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# %%
# Train the model with oversampled data
pipeline.fit(X_resampled, y_resampled)

# %%
# The trained pipeline is used to predict the target variable on the test dataset.
y_pred = pipeline.predict(X_test)

# %% [markdown]
# ## Evaluation

# %%
# 1. Classification Report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=3))

# 2. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# 3. Confusion Matrix (absolute & percentage)
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('Confusion Matrix (Absolute)', fontsize=13)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %%
# Save the trained model pipeline
joblib.dump(pipeline, 'model/rf_classifier_dropout.joblib')

# %%
# Displays the confusion matrix to evaluate classification performance, showing the number of correct and incorrect predictions for each class.
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=pipeline.classes_,
    yticklabels=pipeline.classes_,
    linewidths=0.5,
    cbar=False
)

plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, pad=10)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


