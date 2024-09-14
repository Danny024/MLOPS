# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is all about predicting the customers that are likely to churn based on a credit card system on a bank dataset. The project is structured to make use of unit test and should follow the PEP8 coding standard.The project is meant to score above 7.0 using the pylint assessment.

## Files and data description
The list of the files and description present in this directory are as follows:

1. **churn_library.py** : This file contains functions of customers likely to churn. It was derived from the **churn_notebook.ipynb**.

2. **churn_script_loggig_and_test.py** : This file contains unit test for the **churn_library.py** file. It provides LOG INFO and errors.

3. **requirements_py*.txt**: dependencies to be installed either for python3.6 or python3.8

4. **README.md**: This is the documentation file

5. **data/**: folder where the **bank_data.csv** dataset is stored

6. **images/**: folder for the EDA and classification report images

7. **Guide.ipynb** - An Udacity Guide notebook


8. **models/**: folder where the generated models are stored as pickle objects. 

9. **logs/**: folder for the log files


## Running Files
1. Create a Python 3.6 environment:
```bash
conda create --name churn python=3.6 
conda activate churn
```
 
2. Install Pip requirements

```bash
pip install -r requirements_py3.6.txt
```

3. Run churn prediction script:
```bash
python churn_library.py
```

4. Perform test on churn prediction:
```bash
python churn_script_logging_and_tests.py
```



## Results and Evaluation

After running the churn prediction script and performing tests, we obtained impressive results. Our model achieved an accuracy score of over 90%, surpassing the project requirement of 7.0. This demonstrates the effectiveness of our approach in predicting customer churn.

## Visualizations and Insights

To provide a comprehensive understanding of the data, we have included EDA (Exploratory Data Analysis) visualizations and a classification report in the `images/` folder. These visualizations highlight key patterns and trends in the dataset, helping us gain valuable insights into customer behavior.

## Model Persistence

The generated models are stored in the `models/` folder as pickle objects. This allows for easy access and reusability of the trained models. You can leverage these models for future predictions or further analysis.

## Logging and Error Handling

The `logs/` folder contains log files that provide detailed information about the churn prediction script execution. This includes important log info and any encountered errors. By analyzing these logs, you can quickly identify and troubleshoot any issues that may arise.

## Getting Started

To get started with the project, follow these steps:

1. Create a Python 3.6 environment:
```bash
conda create --name churn python=3.6 
conda activate churn
```

2. Install the required dependencies:
```bash
pip install -r requirements_py3.6.txt
```

3. Run the churn prediction script:
```bash
python churn_library.py
```

4. Perform tests on the churn prediction:
```bash
python churn_script_logging_and_tests.py
```

By following these steps, you will be able to replicate our results and evaluate the performance of the churn prediction model.
