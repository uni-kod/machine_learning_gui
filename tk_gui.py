from tkinter import *

root = Tk()
root.title('Titanic Predictor!')

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


label_page_title = Label(root, text='Titanic machine learning application')
label_page_title.grid(row=0, column=0, padx=10, pady=50, sticky='W')


# Inputs: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title
label_pclass = Label(root, text='Enter Passenger class: [0, 2]:')
entry_pclass = Entry(root, width=5)
label_pclass.grid(row=1, column=0, padx=10, pady=2, sticky='W')
entry_pclass.grid(row=1, column=1, padx=10, pady=2, sticky='W')

label_sex = Label(root, text='Enter Sex: [0, 1]:')
entry_sex = Entry(root, width=5)
label_sex.grid(row=2, column=0, padx=10, pady=2, sticky='W')
entry_sex.grid(row=2, column=1, padx=10, pady=2, sticky='W')

label_age = Label(root, text='Enter Age: [0, inf]:')
entry_age = Entry(root, width=5)
label_age.grid(row=3, column=0, padx=10, pady=2, sticky='W')
entry_age.grid(row=3, column=1, padx=10, pady=2, sticky='W')

label_sibSp = Label(root, text='Enter number of Siblings/Spouses Aboard:  [0, inf]:')
entry_sibSp = Entry(root, width=5)
label_sibSp.grid(row=4, column=0, padx=10, pady=2, sticky='W')
entry_sibSp.grid(row=4, column=1, padx=10, pady=2, sticky='W')

label_parch = Label(root, text='Enter number of Parents/Children Aboard: [0, inf]:')
entry_parch = Entry(root, width=5)
label_parch.grid(row=5, column=0, padx=10, pady=2, sticky='W')
entry_parch.grid(row=5, column=1, padx=10, pady=2, sticky='W')

label_fare = Label(root, text='Enter Fare: [0, inf]')
entry_fare = Entry(root, width=5)
label_fare.grid(row=6, column=0, padx=10, pady=2, sticky='W')
entry_fare.grid(row=6, column=1, padx=10, pady=2, sticky='W')

label_embarked = Label(root, text='Enter port of embarkation: [0, 2]')
entry_embarked = Entry(root, width=5)
label_embarked.grid(row=7, column=0, padx=10, pady=2, sticky='W')
entry_embarked.grid(row=7, column=1, padx=10, pady=2, sticky='W')

label_title = Label(root, text='Enter title [0, 7]:')
entry_title = Entry(root, width=5)
label_title.grid(row=8, column=0, padx=10, pady=2, sticky='W')
entry_title.grid(row=8, column=1, padx=10, pady=2, sticky='W')


# Buttons:
button_train = Button(root, text='Train', command=do_model_train)
button_train.grid(row=9, column=0, padx=10, pady=50, sticky='W')
button_predict = Button(root, text='Predict', command=do_model_predict)
button_predict.grid(row=10, column=0, padx=10, pady=0, sticky='W')

label_prediction = Label(root, text='Prediction: ')
label_prediction.grid(row=11, column=0, padx=10, pady=20, sticky='W')


# Show the window:
root.geometry('500x700')
root.mainloop()
