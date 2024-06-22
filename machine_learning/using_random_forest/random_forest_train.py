#!/usr/bin/env python3

from os.path import dirname
file_path = dirname(__file__)

TRAIN_DATA_FILE   = file_path + '/../train_data/titanic_train.csv'
TEST_DATA_FILE    = file_path + '/../train_data/titanic_test.csv'
OUTPUT_MODEL_FILE = file_path + '/../train_data/titanic_random_forest_1.sav'


debug_train = True


def model_train():
    import pandas as pd
    from .ds_titanic_clean import clean_data_frame
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pickle


    df      = pd.read_csv(TRAIN_DATA_FILE)
    df_test = pd.read_csv(TEST_DATA_FILE)

    ##################################### Data Cleaning ###############################################

    clean_data_frame(df)
    clean_data_frame(df_test)

    ##################################### Machine Learning  ###########################################

    # We chose 'Survived' as y because it is dependent variable.
    y_train = df['Survived']
    # 'Survived' column is already dropped from the original test data file.
    # WRONG: y_test  = df_test['Survived']

    # Most other column are independent variables.
    x_train = df.drop(['Survived'], axis=1)
    # 'Survived' column is already dropped from the original test data file.
    # WRONG: x_test  = df_test.drop(['Survived'], axis=1)
    x_test  = df_test

    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)

    if debug_train:
        # Predict 'y' values from real world 'x' values. 
        y_pred = random_forest.predict(df_test)

        # Calculate the accuracy of our model prediction by comparing the predicted value with the
        # real value.
        acc_random_forest = round(accuracy_score(y_pred, y_train[:len(y_pred)]) * 100, 2)

        print(acc_random_forest)

    # Save the trained model.
    pickle.dump(random_forest, open(OUTPUT_MODEL_FILE, 'wb'))

####################################### Test #####################################################
# model_train()
# from random_forest_predict import model_predict_test
# model_predict_test()
