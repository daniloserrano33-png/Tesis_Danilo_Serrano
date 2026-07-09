# src/loader/data_loader.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """Load loan data from a CSV file.

    This function reads a CSV file from the specified path and returns the data
    as a pandas DataFrame. The low_memory parameter is set to False to ensure
    consistent data type inference across all rows.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame containing loan data.
        None: If the file is not found at the specified path.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"[INFO] Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found at {file_path}")
        return None

def inspect_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial data inspection and create the target variable.

    This function filters the DataFrame to include only loans with definitive
    outcomes by excluding temporary or uncertain statuses such as Current,
    In Grace Period, and Late payments. The function creates a binary target
    variable 'bad_good' where 0 represents fully paid loans (including those
    that do not meet credit policy but were fully paid) and 1 represents
    defaulted, charged-off, or bad loans (including those that do not meet
    credit policy and were charged off). The original 'loan_status' column
    is removed after creating the target variable.

    Args:
        df (pd.DataFrame): The loaded DataFrame containing loan data with
                          a 'loan_status' column.

    Returns:
        pd.DataFrame: The processed DataFrame with the binary target variable
                     'bad_good' and without the 'loan_status' column.
        None: If the input DataFrame is None.
    """
    if df is None:
        return None

    print("\n[INFO] Initial Data Inspection")
    print("\n[INFO] Value counts for 'loan_status':")
    print(df["loan_status"].value_counts())

    print("\n[INFO] Columns in the DataFrame:")
    print(df.columns)

    good_statuses = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid"
    ]

    bad_statuses = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off"
    ]

    exclude_statuses = [
        "Current",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)"
    ]

    df_final = df[df['loan_status'].isin(good_statuses + bad_statuses)].copy()
    df_final['bad_good'] = df_final['loan_status'].apply(
        lambda x: 1 if x in bad_statuses else 0
    )

    print("\n[INFO] Value counts for the target variable 'bad_good':")
    print(df_final['bad_good'].value_counts())

    df_final.drop(columns="loan_status", inplace=True)

    return df_final

def print_dataframe_stats(df: pd.DataFrame):
    """Print basic statistics of the DataFrame.

    This function displays comprehensive information about the DataFrame including
    its shape (rows and columns), data types for each column, count of missing
    values per column, and descriptive statistics for all columns. The descriptive
    statistics include count, mean, standard deviation, minimum, quartiles, and
    maximum values for numerical columns, and count, unique values, top value,
    and frequency for categorical columns.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        None
    """
    print(f"\n[INFO] Prints basic statistics of the DataFrame")
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDescriptive statistics:\n", df.describe(include='all').T) 

def univariate_analysis(df: pd.DataFrame, column: str):
    """Perform univariate analysis on a specified column.

    This function generates appropriate visualizations for a single column based
    on its data type and cardinality. For categorical columns or numerical columns
    with fewer than 10 unique values, a count plot is generated showing the
    frequency of each category. For numerical columns with 10 or more unique values,
    a histogram with kernel density estimation (KDE) curve is generated to show
    the distribution of values.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        column (str): The column name for univariate analysis.

    Returns:
        None
    """
    if column not in df.columns:
        print(f"[WARNING] Column '{column}' not found in DataFrame.")
        return

    plt.figure(figsize=(8, 4))
    if df[column].dtype == 'object' or df[column].nunique() < 10:
        sns.countplot(x=column, data=df)
        plt.title(f'Count Plot of {column}')
    else:
        sns.histplot(df[column].dropna(), bins=30, kde=True)
        plt.title(f'Histogram of {column}')
    
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def bivariate_analysis(df: pd.DataFrame, col1: str, col2: str):
    """Perform bivariate analysis between two specified columns.

    This function generates appropriate visualizations for the relationship between
    two columns based on their data types and cardinality. For categorical columns
    or low-cardinality numerical columns (fewer than 10 unique values), a count
    plot with hue is generated to show the distribution of col1 across different
    values of col2. For high-cardinality numerical columns, a scatter plot is
    generated to visualize the relationship between the two variables.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        col1 (str): The first column name.
        col2 (str): The second column name.

    Returns:
        None
    """
    if col1 not in df.columns or col2 not in df.columns:
        print(f"[WARNING] One or both columns '{col1}', '{col2}' not found in DataFrame.")
        return

    plt.figure(figsize=(8, 4))
    if df[col1].dtype == 'object' or df[col1].nunique() < 10:
        sns.countplot(x=col1, hue=col2, data=df)
        plt.title(f'Count Plot of {col1} by {col2}')
    else:
        sns.scatterplot(x=col1, y=col2, data=df)
        plt.title(f'Scatter Plot of {col1} vs {col2}')
    
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def eda_report(df: pd.DataFrame):
    """Perform comprehensive exploratory data analysis on the DataFrame.

    This function generates a complete EDA report including multiple visualizations
    and statistical summaries. The report includes: DataFrame shape, data types for
    all columns, missing value counts, descriptive statistics for all columns,
    distribution plot of the 'bad_good' target variable (if present), histograms
    for all numerical variables displayed in a grid layout, summary statistics for
    numerical variables, correlation matrix calculations, and a heatmap visualization
    of correlations between numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.

    Returns:
        None
    """
    print("\n[INFO] -------------------- EDA Report --------------------")
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDescriptive statistics:\n", df.describe(include='all'))

    if 'bad_good' in df.columns:
        plt.figure(figsize=(5,3))
        ax = sns.countplot(x='bad_good', data=df)
        plt.title('Target variable distribution (bad_good)', fontsize=10)
        ax.set_xlabel('bad_good', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        plt.show()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        df[num_cols].hist(figsize=(10, 6), bins=30)
        plt.suptitle('Numerical variable histograms', fontsize=10)
        plt.tight_layout()
        plt.show()

    if len(num_cols) > 1:
        print("\n[INFO] Summary numeric variables:")
        print(df[num_cols].describe().T)
        print("\n[INFO] Main correlations:")
        corr = df[num_cols].corr()
        print(corr)
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        sns.heatmap(
            corr,
            annot=False,  
            fmt=".2f",
            cmap='coolwarm',
            cbar_kws={"shrink": 0.7}
        )
        plt.title('Correlation matrix', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    from config import LOAN_FILE
    
    df = load_data(LOAN_FILE)

    df = inspect_and_prepare_data(df)
    print_dataframe_stats(df)
    univariate_analysis(df, 'loan_amnt')
    bivariate_analysis(df, 'loan_amnt', 'bad_good')
    eda_report(df)
 
    if df is not None:
        print("\n[INFO] Processed DataFrame head:")
        print(df.head())