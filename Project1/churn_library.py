# library doc string
"""
Author: Daniel Eneh
Clean code Project
Date : 06 September 2024
"""

# import libraries
import os
import logging as lg
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
#from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV


os.environ['QT_QPA_PLATFORM']='offscreen'



lg.basicConfig(
    filename="logs/customer_turnover.log",
    level=lg.INFO,
    filemode="w",
    format="%(asctime)-15s %(message)s")
logger = lg.getLogger()

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df["Customer_Churn"] = df["Attrition_Flag"].apply(
    lambda val: 0 if val == "Existing Customer" else 1)

    plot_features = ["Customer_Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct", "Data_Heatmap"]
    for feature in plot_features:
        plt.figure(figsize=(25, 10))
        if feature in ("Customer_Churn", "Customer_Age"):
            df[feature].hist()
        elif feature == "Marital_Status":
            df[feature].value_counts(normalize=True).plot(kind='bar')
        elif feature == "Total_Trans_Ct":
            sns.histplot(df[feature], stat='density', kde=True)
        elif feature == "Data_Heatmap":
            sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=2)
        plt.savefig("results/eda/%s.png" % feature)
        plt.close()



def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
      response: string of response name [optional argument that could be used for 
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for column_name in category_lst:
        df[column_name + "_" + response] = df.groupby(column_name)[response].transform('mean')
        
    return df
        
    

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be 
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    x_columns = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"
    ]
                   
    x = df[x_columns].copy()
                   
    y = df[response]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test
                   
   
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    models = {
        "Random_Forest":{
            "train_title": "Random Forest Train",
            "test_title": "Random Forest Test",
            "train_actual": y_train,
            "train_pred" : y_train_preds_rf,
            "test_actual": y_test,
            "test_pred": y_test_preds_rf                        
        },
        "Logistic_Regression":{
            "train_title": "Logistic Regression Train",
            "test_title": "Logistic Regression Test",
            "train_actual": y_train,
            "train_pred": y_train_preds_lr,
            "test_actual": y_test,
            "test_pred": y_test_preds_lr
        }
    }
    for model_name, data in models.items():
        fig, ax = plt.subplots(figsize=(6,6))
        
        report_train = classification_report(data['train_actual'], data['train_pred'])
        report_test = classification_report(data['test_actual'], data['test_pred'])
         
    
    return None

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    #check that model has feature_importances_
    if not hasattr (model.best_estimator_, 'feature_importances_'):
        raise ValueError ("Model does not have feature importance attribute")
    #Incase model has no best_estimator attribute
    if hasattr (model, 'best_estimator_'):
        model_to_use = model.best_estimator_
    else:
        model_to_use = model
    # Extract feature importances and sort them
    feature_importances = model_to_use.best_estimator_.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [X_data.columns[i] for i in sorted_indices]
    
    # Create output directory if one does not exist
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    
    # Plot feature importances
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), feature_importances[sorted_indices])
    plt.xticks(range(X_data.shape[1]), sorted_feature_names, rotation=90)
    plt.tight_layout()  # Optimizes layout
    
    # Save the plot
    plt.savefig(os.path.join(output_pth, "Feature_Importance.jpg"))
    plt.close()
    return None
                   

def train_models(X_train, X_test, y_train, y_test, output_dir="results", model_dir="models"):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Perform GridSearchCV for RandomForest
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
    cv_rfc.fit(X_train, y_train)
    
    # Fit Logistic Regression
    lrc.fit(X_train, y_train)

    # Generate predictions for both models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Store ROC curves plot
    plt.figure(figsize=(20, 10))
    axis = plt.gca()
    
    plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.75)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.75)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Roc_Curves.jpg"))
    plt.close()

    # Store classification report image (assuming classification_report_image function is defined)
    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    # Store feature importance plot
    feature_importance_plot(cv_rfc, X_test, output_dir)

    # Save models
    joblib.dump(cv_rfc.best_estimator_, os.path.join(model_dir, "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(model_dir, "logistic_model.pkl"))
    
    # Output performance metrics
    print("Random Forest - Test Set Performance:")
    print(classification_report(y_test, y_test_preds_rf))
    
    print("\nLogistic Regression - Test Set Performance:")
    print(classification_report(y_test, y_test_preds_lr))

#lg.basicConfig(level=logging.INFO)
#logger = lg.getLogger()

# Constants for categorical columns and response
CATEGORY_COLS = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
RESPONSE_COL = "Churn"
DATA_PATH = "data/bank_data.csv"

def main(data_path=DATA_PATH):
    try:
        logger.info("Importing data from %s", data_path)
        df_raw = import_data(data_path)
    except FileNotFoundError as e:
        logger.error("Error: Data file not found - %s", e)
        return
    except Exception as error:
        logger.error("Error occurred during data import - %s", error)
        return

    try:
        logger.info("Performing EDA")
        perform_eda(df_raw)
    except Exception as error:
        logger.error("Error during EDA - %s", error)
        return

    try:
        logger.info("Encoding categorical columns")
        df_encoded = encoder_helper(df_raw, category_lst=CATEGORY_COLS, response=RESPONSE_COL)
    except Exception as error:
        logger.error("Error during encoding - %s", error)
        return

    try:
        logger.info("Performing feature engineering and splitting data")
        X_train, X_test, y_train, y_test = perform_feature_engineering(df_encoded, response=RESPONSE_COL)
    except Exception as error:
        logger.error("Error during feature engineering - %s", error)
        return

    try:
        logger.info("Training models and saving results")
        train_models(X_train, X_test, y_train, y_test)
    except Exception as error:
        logger.error("Error during model training - %s", error)
        return

if __name__ == "__main__":
    main()