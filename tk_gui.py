from tkinter import *

root = Tk()

def do_model_train():
    from machine_learning.using_neural_network.neural_network_train import model_train
    #from machine_learning.using_random_forest.random_forest_train import model_train
    model_train()


def do_model_predict():
    from machine_learning.using_neural_network.neural_network_predict import model_predict
    #from machine_learning.using_random_forest.random_forest_predict import model_predict

    pclass   = int(entry_pclass.get()) 
    sex      = int(entry_sex.get()) 
    age      = int(entry_age.get()) 
    sibSp    = int(entry_sibSp.get()) 
    parch    = int(entry_parch.get()) 
    fare     = int(entry_fare.get()) 
    embarked = int(entry_embarked.get()) 
    title    = int(entry_title.get()) 


    prediction = model_predict(pclass, sex, age, sibSp, parch, fare, embarked, title)
    if prediction < 0.5:
        prediction = 'Not survived'
    else:
        prediction = 'Survived'
    label_prediction.config(text='Prediction: {}'.format(prediction))


label_title = Label(root, text='Titanic machine learning application')

# Inputs: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title
entry_pclass   = Entry(root)
entry_pclass.insert(0, 'pclass')
entry_sex      = Entry(root)
entry_sex.insert(0, 'sex')
entry_age      = Entry(root)
entry_age.insert(0, 'age')
entry_sibSp    = Entry(root)
entry_sibSp.insert(0, 'sibSp')
entry_parch    = Entry(root)
entry_parch.insert(0, 'parch')
entry_fare     = Entry(root)
entry_fare.insert(0, 'fare')
entry_embarked = Entry(root)
entry_embarked.insert(0, 'embarked')
entry_title    = Entry(root)
entry_title.insert(0, 'title')

# Buttons:
button_train = Button(root, text='Train', command=do_model_train)
button_predict = Button(root, text='Predict', command=do_model_predict)

label_prediction = Label(root, text='Prediction: ')

# Pack all widgets:
label_title.pack()

entry_pclass.pack()
entry_sex.pack()
entry_age.pack()
entry_sibSp.pack()
entry_parch.pack()
entry_fare.pack()
entry_embarked.pack()
entry_title.pack()

button_train.pack()
button_predict.pack()

label_prediction.pack()

# Show the window:
root.geometry('500x500')
root.mainloop()
