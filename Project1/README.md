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
