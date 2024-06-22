#!/usr/bin/env python3

from os.path import dirname
file_path = dirname(__file__)

OUTPUT_MODEL_FILE = file_path + '/../train_data/titanic_random_forest_1.sav'


####################################### Prediction ###############################################

#print(df.head)
# Result: Survived.
# Input: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title
def model_predict_array(x_input):
    import pickle

    random_forest = pickle.load(open(OUTPUT_MODEL_FILE, 'rb'))
    prediction = random_forest.predict(x_input)
    # Note: random_forest.predict() returns [1], or [0]
    if prediction == [0]:
        prediction = 0
    else:
        prediction = 1
    return prediction

def model_predict(pclass, sex, age, sibSp, parch, fare, embarked, title):
    x_input = [[pclass, sex, age, sibSp, parch, fare, embarked, title]]
    prediction = model_predict_array(x_input)
    return prediction

def model_predict_test():
    # This data results in: 1 --> survived.
    x_example = [[1, 1, 11, 1, 1, 19, 1, 1]]
    # This data results in: 0 --> not survived.
    # x_example = [[0, 0, 60, 0, 0, 10, 1, 1]]

    prediction = model_predict_array(x_example)
    print(x_example)
    print(prediction)
    if prediction < 0.5:
        prediction = 'Not survived'
    else:
        prediction = 'Survived'
    print(prediction)

# Test the prediction model.
# model_predict_test()
