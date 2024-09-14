'''
Name : Daniel Eneh
Date : 8th September 2024
'''
import os
import glob
import logging
import pytest
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/test_log.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# Fixtures

@pytest.fixture(scope="module")
def dataframe_raw():
    """
    Fixture for loading the raw dataset, executed once for all tests.
    """
    try:
        df = import_data('data/bank_data.csv')
        logging.info("Data loaded successfully for tests")
        return df
    except FileNotFoundError as err:
        logging.error("Error loading data: File not found")
        raise err


@pytest.fixture(scope="module")
def dataframe_encoded(dataframe_raw):
    """
    Fixture for encoded dataset, applied on the raw data.
    """
    try:
        df = encoder_helper(dataframe_raw,
                            ["Gender",
                             "Education_Level",
                             "Marital_Status",
                             "Income_Category",
                             "Card_Category"],
                            'Churn')
        logging.info("Data encoding applied successfully for tests")
        return df
    except KeyError as err:
        logging.error("Error during encoding: Incorrect column names")
        raise err

@pytest.fixture(scope="module")
def dataframe_split(dataframe_encoded):
    """
    Fixture for performing feature engineering, splitting the data into train and test sets.
    """
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            dataframe_encoded, 'Churn')
        logging.info("Feature engineering and data splitting successful")
        return X_train, X_test, y_train, y_test
    except ValueError as err:
        logging.error("Error during feature engineering: %s", err)
        raise err

# Test functions
def test_import_data(dataframe_raw):
    """
    Test that checks if the data import function works properly.
    """
    try:
        assert dataframe_raw.shape[0] > 0, "The dataset should contain rows"
        assert dataframe_raw.shape[1] > 0, "The dataset should contain columns"
        logging.info("Data import test passed")
    except AssertionError as err:
        logging.error("Data import test failed")
        raise err


def test_perform_eda(dataframe_raw):
    """
    Test that checks if the EDA function runs and generates the expected output files.
    """
    try:
        perform_eda(dataframe_raw)
        logging.info("EDA function executed successfully")
    except Exception as err:
        logging.error("EDA function test failed: %s", err)
        raise err

    # Validate the creation of EDA images
    eda_images = [
        'Customer_Churn.png',
        'Customer_Age.png',
        'Marital_Status.png',
        'Total_Trans_Ct.png',
        'Data_Heatmap.png']
    for image in eda_images:
        try:
            assert os.path.isfile(f'results/eda/{image}')
            logging.info(f"{image} file created successfully")
        except AssertionError:
            logging.error(f"EDA image file {image} was not created")
            raise


def test_encoder_helper(dataframe_encoded):
    """
    Test that checks if the encoder_helper function properly encodes the specified columns.
    """
    encoded_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    try:
        for col in encoded_columns:
            assert col in dataframe_encoded.columns, f"Column {col} not found in the encoded dataframe"
        logging.info("Encoder helper test passed")
    except AssertionError as err:
        logging.error("Encoder helper test failed: %s", err)
        raise err


def test_perform_feature_engineering(dataframe_split):
    """
    Test to check if the feature engineering and data splitting is done correctly.
    """
    X_train, X_test, y_train, y_test = dataframe_split
    try:
        assert X_train.shape[0] > 0, "X_train should not be empty"
        assert X_test.shape[0] > 0, "X_test should not be empty"
        assert y_train.shape[0] > 0, "y_train should not be empty"
        assert y_test.shape[0] > 0, "y_test should not be empty"
        logging.info("Feature engineering test passed")
    except AssertionError as err:
        logging.error("Feature engineering test failed: %s", err)
        raise err


def test_train_models(dataframe_split):
    """
    Test to check if the models are trained and saved correctly.
    """
    X_train, X_test, y_train, y_test = dataframe_split
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Model training function executed successfully")
    except Exception as err:
        logging.error(f"Model training test failed: {err}")
        raise err

    # Check for model files been created
    model_files = ['models/rfc_model.pkl', 'models/logistic_model.pkl']
    for model_file in model_files:
        try:
            assert os.path.isfile(model_file), f"{model_file} not found"
            logging.info(f"Model file {model_file} created successfully")
        except AssertionError as err:
            logging.error(f"Model file {model_file} not found")
            raise err

    # Check for ROC curve plot is created
    try:
        assert os.path.isfile(
            'results/Roc_Curves.jpg'), "ROC curve plot not found"
        logging.info("ROC curve plot created successfully")
    except AssertionError as err:
        logging.error("ROC curve plot not found")
        raise err


if __name__ == "__main__":
    # Clean path
    for dir_path in ['logs', 'results/eda', 'results', 'models']:
        files = glob.glob(f'{dir_path}/*')
        for file in files:
            os.remove(file)

    # Run the tests
    pytest.main()
