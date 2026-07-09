# src/preprocess/preprocessing.py

import pandas as pd
import numpy as np
import scipy.stats as stats

from src.loader import data_loader as dl

def handle_missing_values(df, missing_percentage_threshold=30):
    """Identify and handle missing values in the DataFrame.

    This function calculates the percentage of missing values for each column
    and drops columns that exceed the specified missing percentage threshold.
    The function displays missing value statistics sorted by percentage and
    removes columns above the threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        missing_percentage_threshold (int): The threshold percentage above which
                                            columns will be dropped.
    
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame after dropping high-missing columns.
            - list: List of column names that were dropped due to exceeding
                   the missing percentage threshold.
    """
    print("\n[INFO] Handling Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_percentages = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage': missing_percentages})
    print(missing_info.sort_values(by='Missing Percentage', ascending=False))

    cols_to_drop = missing_info[missing_info['Missing Percentage'] > missing_percentage_threshold].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with > {missing_percentage_threshold}% missing values: {cols_to_drop}")
    else:
        print(f"No columns with > {missing_percentage_threshold}% missing values to drop.")

    return df, cols_to_drop

def impute_missing_values(df):
    """Impute missing values for numerical and categorical features.

    This function fills missing values in numerical columns using the median
    value and fills missing values in categorical columns using the mode value.
    The function processes each column with missing data and applies the
    appropriate imputation strategy based on data type.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame after imputing all missing values.
    """
    print("\n[INFO] Imputing Missing Values")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Imputed missing values in numerical column '{col}' with median: {median_value}")
            elif df[col].dtype == 'object':
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print(f"Imputed missing values in categorical column '{col}' with mode: {mode_value}")
            else:
                print(f"Column '{col}' has unsupported data type for imputation: {df[col].dtype}")

    return df

def convert_specific_categorical_to_numeric(df):
    """Convert specific categorical columns to numeric representations.

    This function performs domain-specific conversions for loan data columns.
    The 'term' column is mapped to numeric months (36 or 60). The 'emp_length'
    column is extracted to numeric years. The 'int_rate' column is converted
    from percentage string to decimal. The 'revol_util' column is converted
    from percentage string to decimal. Original categorical columns are dropped
    after conversion.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with converted numeric columns replacing
                      original categorical columns.
    """
    print("\n[INFO] Converting specific categorical columns to numeric")
    if 'term' in df.columns:
        df['term_num'] = df['term'].map({' 36 months': 36, ' 60 months': 60})
        df.drop(columns=['term'], inplace=True)
        print("Converted 'term' to 'term_num'.")
    else:
        print("'term' column not found.")

    if 'emp_length' in df.columns:
        df['emp_length_num'] = df['emp_length'].astype(str).str.extract(r'(\d+)')[0]
        df['emp_length_num'] = pd.to_numeric(df['emp_length_num'], errors='coerce')
        df.drop(columns=['emp_length'], inplace=True)
        print("Converted 'emp_length' to 'emp_length_num'.")
    else:
        print("'emp_length' column not found.")

    if 'int_rate' in df.columns:
        df['int_rate_decimal'] = (
            df['int_rate']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
            .replace('', np.nan)
            .astype(float) / 100
        )
        df.drop(columns=['int_rate'], inplace=True)
        print("Converted 'int_rate' to 'int_rate_decimal'.")
    else:
        print("'int_rate' column not found.")

    if 'revol_util' in df.columns:
        df['revol_util_decimal'] = (
            df['revol_util']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
            .replace('', np.nan)
            .astype(float) / 100
        )
        df.drop(columns=['revol_util'], inplace=True)
        print("Converted 'revol_util' to 'revol_util_decimal'.")
    else:
        print("'revol_util' column not found.")

    return df

def drop_highly_correlated_features(df, correlation_threshold=0.7):
    """Identify and drop highly correlated numerical features.

    This function calculates the Pearson correlation matrix for numerical
    features, excluding the target variable 'bad_good'. Features with absolute
    correlation coefficients above the specified threshold are identified using
    the upper triangle of the correlation matrix and subsequently dropped from
    the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        correlation_threshold (float): The absolute correlation threshold above
                                       which features are dropped.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame after dropping highly correlated features.
            - list: List of column names that were dropped due to high correlation.
    """
    print(f"\n[INFO] Dropping highly correlated numerical features (threshold > {correlation_threshold})")
    numeric_df = df.select_dtypes(include=np.number).drop(columns=['bad_good'], errors='ignore')

    correlation_matrix = numeric_df.corr().abs()

    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    to_drop_correlated = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    if to_drop_correlated:
        df.drop(columns=to_drop_correlated, inplace=True, errors='ignore')
        print(f"Dropped highly correlated columns: {to_drop_correlated}")
    else:
        print("No highly correlated columns to drop.")

    return df, to_drop_correlated

def drop_low_ks_features(df, target_col='bad_good', ks_threshold=0.03):
    """Identify and drop numerical features with low Kolmogorov-Smirnov statistics.

    This function calculates the two-sample Kolmogorov-Smirnov statistic for each
    numerical feature with respect to the binary target variable. The KS statistic
    measures the maximum difference between the cumulative distributions of the
    feature values for the two target classes. Features with KS statistics below
    the specified threshold are dropped from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the binary target variable column.
        ks_threshold (float): The minimum KS statistic threshold below which
                             features are dropped.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame after dropping low KS features.
            - list: List of column names that were dropped due to low KS statistics.
    """
    print(f"\n[INFO] Dropping numerical features with low KS statistics (threshold < {ks_threshold})")
    numeric_df = df.select_dtypes(include=np.number)

    def calculate_ks(df, feature, target):
        df_filtered = df[df[target].isin([0, 1])]
        group_0 = df_filtered[df_filtered[target] == 0][feature].dropna()
        group_1 = df_filtered[df_filtered[target] == 1][feature].dropna()

        if len(group_0) > 0 and len(group_1) > 0:
            ks_stat, p_value = stats.ks_2samp(group_0, group_1)
            return ks_stat
        return 0.0

    ks_results = []
    for col in numeric_df.columns:
        if col != target_col:
            ks_stat = calculate_ks(numeric_df, col, target_col)
            ks_results.append({'Variable': col, 'KS Statistic': ks_stat})

    ks_df = pd.DataFrame(ks_results).sort_values(by='KS Statistic', ascending=False)
    print("KS Statistics for numerical features:")
    print(ks_df)

    cols_to_drop = ks_df[ks_df['KS Statistic'] < ks_threshold].Variable.tolist()

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        print(f"Dropped low KS columns: {cols_to_drop}")
    else:
         print("No low KS columns to drop.")

    return df, cols_to_drop

def drop_low_variance_features(df, variance_threshold=0.01):
    """Identify and drop numerical features with low variance.

    This function calculates the variance for each numerical feature in the
    DataFrame. Features with variance values below the specified threshold
    are considered to have insufficient variability and are dropped.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variance_threshold (float): The minimum variance threshold below which
                                    features are dropped.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame after dropping low variance features.
            - list: List of column names that were dropped due to low variance.
    """
    print(f"\n[INFO] Dropping numerical features with low variance (threshold < {variance_threshold})")
    numeric_df = df.select_dtypes(include=np.number)

    variances = numeric_df.var()
    low_variance_cols = variances[variances < variance_threshold].index.tolist()

    if low_variance_cols:
        df.drop(columns=low_variance_cols, inplace=True, errors='ignore')
        print(f"Dropped low variance columns: {low_variance_cols}")
    else:
        print("No low variance columns to drop.")

    return df, low_variance_cols

def drop_high_variance_features(df, variance_threshold=1000):
    """Identify and drop numerical features with high variance.

    This function calculates the variance for each numerical feature in the
    DataFrame. Features with variance values above the specified threshold
    are considered to have excessive variability and are dropped.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variance_threshold (float): The maximum variance threshold above which
                                    features are dropped.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame after dropping high variance features.
            - list: List of column names that were dropped due to high variance.
    """
    print(f"\n[INFO] Dropping numerical features with high variance (threshold > {variance_threshold})")
    numeric_df = df.select_dtypes(include=np.number)

    variances = numeric_df.var()
    high_variance_cols = variances[variances > variance_threshold].index.tolist()

    if high_variance_cols:
        df.drop(columns=high_variance_cols, inplace=True, errors='ignore')
        print(f"Dropped high variance columns: {high_variance_cols}")
    else:
        print("No high variance columns to drop.")

    return df, high_variance_cols

def drop_descriptive_features(df):
    """Drop descriptive text features and unnamed columns.

    This function removes predefined columns that are identified as having low
    relevance for predictive modeling. These include identifier columns (id,
    member_id), location information (zip_code, addr_state), text descriptions
    (title, desc, emp_title), temporal columns (issue_d, earliest_cr_line,
    last_pymnt_d, last_credit_pull_d, next_pymnt_d), and administrative fields
    (url, policy_code, pymnt_plan, application_type, sub_grade). Additionally,
    any columns with names starting with 'Unnamed' are removed.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame after dropping descriptive features.
            - list: List of column names that were dropped.
    """
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
    cols_to_drop = ["id","member_id","zip_code", "sub_grade", "emp_title", "issue_d", "url", "title", "addr_state","policy_code", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d","pymnt_plan","next_pymnt_d","application_type","emp_title","sub_grade","desc"]
    cols_to_drop += unnamed_cols
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"\n[INFO] Dropped initial columns: {cols_to_drop}")
    print(f"Remaining columns after initial dropping: {df.columns.tolist()}")
    return df, cols_to_drop

def drop_low_iv_categorical_features(df, categorical_cols, target='bad_good', iv_threshold=0.02):
    """Calculate Information Value for categorical features and drop low IV features.

    This function computes the Information Value (IV) for each specified
    categorical feature with respect to the binary target variable. IV is a
    measure of the predictive power of a feature for a binary classification
    problem. Features with IV values below the specified threshold are considered
    to have insufficient predictive power and are dropped from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_cols (list): List of categorical column names to evaluate.
        target (str): The name of the binary target column (1 for bad, 0 for good).
        iv_threshold (float): The minimum IV threshold below which features are dropped.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame after dropping low-IV categorical features.
            - list: List of column names that were dropped due to low IV.
            - pd.DataFrame: DataFrame containing IV values for each evaluated feature,
                          sorted by IV in descending order.
    """
    iv_results = []
    for col in categorical_cols:
        iv = calculate_categorical_iv(df, col, target)
        iv_results.append({'variable': col, 'iv': iv})

    iv_df = pd.DataFrame(iv_results).sort_values(by='iv', ascending=False)
    print("[INFO] IV values for categorical features:")
    print(iv_df)

    cols_to_drop = iv_df[iv_df['iv'] < iv_threshold]['variable'].tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
        print(f"Dropped categorical columns with IV < {iv_threshold}: {cols_to_drop}")
    else:
        print("No categorical columns with low IV to drop.")

    return df, cols_to_drop, iv_df

def calculate_categorical_iv(df, feature, target):
    """Calculate the Information Value for a categorical feature.

    This function computes the IV metric by first filling missing values with
    the string "MISSING", then grouping data by the feature to calculate counts
    of total samples, bad samples, and good samples for each category. The
    distribution of bad and good samples is computed, followed by the Weight
    of Evidence (WoE) for each category. The IV is calculated as the sum of
    (distribution_good - distribution_bad) * WoE across all categories.

    Args:
        df (pd.DataFrame): The input DataFrame containing the feature and target columns.
        feature (str): The name of the categorical feature column.
        target (str): The name of the binary target column (1 for bad, 0 for good).

    Returns:
        float: The total Information Value for the specified feature.
    """
    df = df.copy()
    df[feature] = df[feature].fillna("MISSING")
    summary_vars = (
        df.groupby(feature)[target]
        .agg(['count', 'sum'])
        .rename(columns={'count': 'total', 'sum': 'bad'})
    )
    summary_vars['good'] = summary_vars['total'] - summary_vars['bad']
    total_bad = summary_vars['bad'].sum()
    total_good = summary_vars['good'].sum()
    summary_vars['dist_bad'] = summary_vars['bad'] / total_bad if total_bad > 0 else 0
    summary_vars['dist_good'] = summary_vars['good'] / total_good if total_good > 0 else 0
    summary_vars['woe'] = np.log((summary_vars['dist_good'] / summary_vars['dist_bad']).replace(0, np.nan))
    summary_vars['woe'] = summary_vars['woe'].replace([np.inf, -np.inf], 0).fillna(0)
    summary_vars['iv'] = (summary_vars['dist_good'] - summary_vars['dist_bad']) * summary_vars['woe']
    iv_total = summary_vars['iv'].sum()
    return iv_total

def encode_categorical_features(df, target_col='bad_good'):
    """Encode remaining categorical features using one-hot encoding.

    This function applies one-hot encoding to all categorical columns in the
    DataFrame using pandas get_dummies. The target variable is temporarily
    removed before encoding and added back after the transformation to preserve
    its original values. Each unique category value in a categorical column is
    converted into a separate binary column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target variable column to preserve.

    Returns:
        pd.DataFrame: The DataFrame with categorical features one-hot encoded
                      and the target column restored.
    """
    print("\n[INFO] Encoding categorical features")
    df_encoded = df.drop(columns=[target_col], errors='ignore')
    target_data = df[target_col]

    categorical_cols = df_encoded.select_dtypes(include=['object']).columns

    if categorical_cols.empty:
        print("No categorical columns to encode.")
        if target_col not in df_encoded.columns and target_col in df.columns:
             df_encoded[target_col] = target_data
        return df_encoded

    print(f"Categorical columns to encode: {list(categorical_cols)}")
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, dummy_na=False)

    df_encoded[target_col] = target_data

    print("DataFrame after encoding categorical features:")
    print(df_encoded.head())

    return df_encoded

if __name__ == '__main__':
    
    from config import LOAN_FILE
    
    df = dl.load_data(LOAN_FILE)
    df = dl.inspect_and_prepare_data(df)
    dl.print_dataframe_stats(df)

    dl.eda_report(df)

    print("[INFO] Original DataFrame head:")
    print(df.head())
    print("\n[INFO] Original DataFrame info:")
    df.info()

    df_processed, dropped_missing = handle_missing_values(df.copy(), missing_percentage_threshold=30)
    df_processed = convert_specific_categorical_to_numeric(df_processed)
    df_processed = impute_missing_values(df_processed)
    df_processed, dropped_low_variance = drop_low_variance_features(df_processed, variance_threshold=0.01)
    df_processed, dropped_descriptive = drop_descriptive_features(df_processed)
    df_processed, dropped_correlated = drop_highly_correlated_features(df_processed, correlation_threshold=0.6)
    df_processed, dropped_low_ks = drop_low_ks_features(df_processed, target_col='bad_good', ks_threshold=0.03)
    df_processed, dropped_low_iv, iv_df = drop_low_iv_categorical_features(df_processed, df_processed.select_dtypes(include=['object']).columns.tolist(), target='bad_good', iv_threshold=0.02)

    print("\n[INFO] Final Processed DataFrame head:")
    print(df_processed.head())
    print("\n[INFO] Final Processed DataFrame info:")
    df_processed.info()