#!/usr/bin/env python3

from os.path import dirname
file_path = dirname(__file__)

TRAIN_DATA_FILE   = file_path + '/../train_data/titanic_train.csv'
TEST_DATA_FILE    = file_path + '/../train_data/titanic_test.csv'
OUTPUT_MODEL_FILE = file_path + '/../train_data/titanic_neural_network_1.keras'


debug_train = True


def model_train():
    import pandas as pd
    from .ds_titanic_clean import clean_data_frame
    from keras.api.models import Sequential
    from keras.api.layers import Dense, Dropout, Input
    import matplotlib.pyplot as plt

    df      = pd.read_csv(TRAIN_DATA_FILE)
    df_test = pd.read_csv(TEST_DATA_FILE)

    ##################################### Data Cleaning ###############################################

    clean_data_frame(df)
    clean_data_frame(df_test)

    ##################################### Deep Learning  #############################################
    # Hyperparameters:
    epochs = 100

    # We chose 'Survived' as y because it is dependent variable.
    y_train = df['Survived']
    # 'Survived' column is already dropped from the original test data file.
    # WRONG: y_test  = df_test['Survived']

    # Most other column are independent variables.
    x_train = df.drop(['Survived'], axis=1)
    # 'Survived' column is already dropped from the original test data file.
    # WRONG: x_test  = df_test.drop(['Survived'], axis=1)
    x_test  = df_test

    # Create a placeholder for the neural network.
    model = Sequential()

    model.add(Input(shape=(8, )))

    # Add the first hidden layer (dense layer) to the neural network:
    #  - Add 32 nodes to our neural network.
    #  - Set the activation function to 'Relu', because we dont want the vanishing gradient in the
    #    'Sigmoid' activation function.
    #  - Set the input layer shape.
    model.add(Dense(32, activation='relu'))
    # We will add a dropout layer between every fully connected layer, hopefully that will at least
    # reduce the difference between the training accuracy and the validation accuracy.
    # Dropout is regularization technique (one of the hyperparameters) that is used to reduce
    # overfitting.
    model.add(Dropout(0.2))

    # Now we have fully connected first layer, we will add more fully connected layers.
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    # Now we have set all the hidden layers, we will add the output layer, and because the output
    # for this example is only one (survived) and the output values is only 0 or 1 (survived or not)
    # we will use the 'Sigmoid' activation function.
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 6% of our data are used for validation.
    model_train = model.fit(x_train, y_train, epochs=epochs, batch_size=50, verbose=0, validation_split=0.06)

    if debug_train:
        # Visualize how the accuracy changes with the number of epochs.
        plt.plot(model_train.history['accuracy'], label='train')
        plt.plot(model_train.history['val_accuracy'], label='test')
        plt.plot(model_train.history['val_loss'], label='loss')
        plt.title('Model accuracy')
        plt.xlabel('Epoch number')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    # Save the trained model.
    model.save(OUTPUT_MODEL_FILE)

####################################### Test #####################################################
# model_train()
# from neural_network_predict import model_predict_test
# model_predict_test()
