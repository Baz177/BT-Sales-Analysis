import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import dataframe_image as dfi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# Ensure static directory exists
static_dir = os.path.join(os.getcwd(), 'static')
os.makedirs(static_dir, exist_ok=True)

# Load the dataset
raw = pd.read_csv(r"C:\Users\bkt29\OneDrive\Desktop\MLE_AI\datasets\BT_cust_interactions.csv")
print("Original columns:", raw.columns.tolist())

# Trim whitespace from column names
raw.columns = [col.strip() for col in raw.columns]


# Remove whitespace from string columns
def remove_whitespace_df(dataframe):
    object_columns = dataframe.select_dtypes(include=['object']).columns
    for col in object_columns:
        dataframe[col] = dataframe[col].apply(lambda x: ' '.join(x.split()) if isinstance(x, str) else x)
    return dataframe

raw = remove_whitespace_df(raw)
raw.drop(index=218, inplace=True)
df = raw[['Day', 'Ethnicity', 'In_Market_Reason', 'MaxCare', 'Sex', 'Credit_Application', 'PreQualification', 'Transfered_Car_In', 'Purchased_Car']]


purchased_cars = df['Purchased_Car'].value_counts()
print("Purchased Cars Distribution:\n", purchased_cars)

# Collecting basic statistics 
def plot_purchased_cars_distribution(dataframe):
    '''
    This function plots the distribution of purchased cars based on categorical variables.
    '''
    pruchase_df = dataframe[dataframe['Purchased_Car'] == 1]
    no_purchase_df = dataframe[dataframe['Purchased_Car'] == 0]
    for col in ['Day', 'Ethnicity', 'In_Market_Reason', 'Sex']:
        (dataframe[col].value_counts()/100).plot(kind='bar', title=col, figsize=(10, 6), alpha=0.7, color='blue')
        plt.xlabel(col)
        plt.ylabel('Percentage')
        plt.title(f'Distribution of Purchased Cars by {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, f'plot_{col}.png'))
        plt.show()
    return None 


def cat_to_num(df):
    '''
    This function converts categorical variables to numerical variables.
    '''
    # Convert categorical variables to numerical
    print("DataFrame before one-hot encoding:\n", df.columns.tolist())
    cat_cols = ['Day', 'Ethnicity', 'In_Market_Reason', 'MaxCare', 'Sex']
    df_one_hot_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(int)
    print("One-hot encoded columns:", df_one_hot_enc.columns.tolist())
    return df_one_hot_enc

# Testing significance of categorical variables
def t_test(df):
    # Reset index to ensure uniqueness
    df = df.reset_index(drop=True)
    
    # Initialize lists to store results
    t_test_values = []
    p_values = []
    columns = []
    
    # Perform t-tests for each relevant column (excluding 'Purchased_Car')
    cols = df.columns.tolist()
    cols.remove('Purchased_Car')
    print ("Columns for t-test:", cols)
    for col in cols: 
        purchased = df[df['Purchased_Car'] == 1][col]
        not_purchased = df[df['Purchased_Car'] == 0][col]
            
        # Perform t-test (handle potential NaN or empty groups)
        if len(purchased) > 0 and len(not_purchased) > 0:  # Ensure groups are non-empty
            t_stat, p_value = stats.ttest_ind(purchased, not_purchased, nan_policy='omit')
            t_test_values.append(round(t_stat, 2))
            p_values.append(round(p_value, 4))
            columns.append(col)
    
    # Create DataFrame with results
    df_t_test = pd.DataFrame({
        't_stat': t_test_values,
        'p_value': p_values,
        'significant': [p < 0.05 for p in p_values]
    }, index=columns)
    
    # Convert 'significant' column to 'Yes'/'No'
    df_t_test['significant'] = df_t_test['significant'].replace({True: 'Yes', False: 'No'})
    return df_t_test

def process_significance(df):
    '''
    This function processes one-hot encoding for categorical variables.
    '''
    # Adding the columns back to original columns 
    original = {}
    for item in ['Day', 'Ethnicity', 'In_Market_Reason', 'MaxCare', 'Sex']:
        sum_t_stat = 0
        sum_p_value = 0
        for idx in df.index:
            if idx.startswith(item):
                sum_t_stat += df.loc[idx, 't_stat']
                sum_p_value += df.loc[idx, 'p_value']
        original[item] = sum_t_stat, sum_p_value
        #print("original data:", original)
    
    for idx in ['Credit_Application', 'PreQualification', 'Transfered_Car_In']:
        original[idx] = df.loc[idx, 't_stat'], df.loc[idx, 'p_value']
    df = pd.DataFrame(original, index=['t_stat', 'p_value']).T
    df['significant'] = df['p_value'].apply(lambda x: 'Yes' if x < 0.05 else 'No')
    return df

def highlight_significant(s):
    '''
    Highlights cells in a row based on the 'significant' column value.
    - Lightcoral for 'Yes' (highly significant, p_value < 0.05)
    - No color for 'No' (not significant, p_value >= 0.05)
    Parameters:
        s (pandas.Series): A row of the DataFrame containing the 'significant' column.
    Returns:
        list: A list of CSS styles (same length as the number of columns) to apply to the row.
    '''
    if 'significant' not in s:
        return [''] * len(s)
    
    if s['significant'] == 'Yes':
        return ['background-color: lightcoral'] * len(s)
    else:
        return [''] * len(s)

def feature_importance(df):
    '''
    This function calculates the feature importance using Random Forest.
    '''
    # Define features and target variable
    print("DataFrame before feature importance:\n", df.columns.tolist())
    rf_df = df.copy()
   
    y = rf_df['Purchased_Car']
    X = rf_df.drop(columns=['Purchased_Car'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    rf.fit(X_train, y_train)

    # Get feature importance
    feature_importances = rf.feature_importances_
    
    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    
    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    return feature_importance_df

def plot_feature_importance(df):
    '''
    This function plots the feature importance.
    '''
    # Adding the columns back to original columns 
    original = {}
    for item in ['Day', 'Ethnicity', 'In_Market_Reason', 'MaxCare', 'Sex']:
        sum_feature = 0
        for idx, row in df.iterrows():
            if row['Feature'].startswith(item):
                sum_feature += row['Importance']
        original[item] = sum_feature
    
    for idx, row in df.iterrows():
        if row['Feature'] in ['Credit_Application', 'PreQualification', 'Transfered_Car_In']:
            original[row['Feature']] = row['Importance']

    importance_df_new = pd.DataFrame(original, index=['Importance']).T
    importance_df_new.drop('MaxCare', axis=0, inplace=True)
    print("New Feature Importance DataFrame:\n", importance_df_new)
    importance_df_new = importance_df_new.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x = 'index', y = 'Importance', data=importance_df_new.reset_index(), palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'feature_importance.png'))
    plt.show()
    return None


# Plotting the distribution of purchased cars
plot_purchased_cars_distribution(df)

# Converting categorical variables to numerical
df_one_hot_enc = cat_to_num(df)

# Feature importance using Random Forest
df = feature_importance(df_one_hot_enc)

# Saving the feature importance plot
plot_feature_importance(df)

# Performing t-tests
df_t_test = t_test(df_one_hot_enc)
df_significance = process_significance(df_t_test)
df_original_cols = process_significance(df_significance)
df_significance_styled = df_original_cols.style.apply(highlight_significant, axis=1).format({'P_value': '{:.3f}'})

# Save the styled DataFrame as an image
dfi.export(df_significance_styled, 'static/styled_dataframe.png')
 