#!/usr/bin/env python3

from os.path import dirname
file_path = dirname(__file__)

OUTPUT_MODEL_FILE = file_path + '/../train_data/titanic_neural_network_1.keras'


####################################### Prediction ###############################################

#print(df.head)
# Result: Survived.
# Input: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title
def model_predict_array(x_input):
    from keras.api.models import load_model

    model_predict = load_model(OUTPUT_MODEL_FILE)
    prediction = model_predict.predict(x_input)
    return prediction

def model_predict(pclass, sex, age, sibSp, parch, fare, embarked, title):
    import numpy as np
    x_input = np.array([[pclass, sex, age, sibSp, parch, fare, embarked, title]])
    prediction = model_predict_array(x_input)
    return prediction

def model_predict_test():
    import numpy as np

    # This data results in: 0.8991854 --> survived.
    x_example = np.array([[1, 1, 11, 1, 1, 19, 1, 1]])
    # This data results in: 0.12963091 --> not survived.
    # x_example = np.array([[0, 0, 60, 0, 0, 10, 1, 1]])

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
