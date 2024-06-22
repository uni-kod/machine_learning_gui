##################################### Data Cleaning ###############################################

# Extract the titles from all names.
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()

def shorted_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'the Countess', 'Dona', 'Lady', 'Sir']:
        return 'Royalty'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title

def clean_data_frame(df):
    # titles = set(x for x in df['Name'].map(lambda x: get_title(x)))

    # Create a new column 'Title', from the extracted titles from 'Name' column.
    df['Title'] = df['Name'].map(lambda x: get_title(x))
    # Change some titles to a common title.
    df['Title'] = df.apply(shorted_titles, axis=1)

    # Fill the empry values in 'Age', 'Fare' and 'Embarked' columns.
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)

    # Remove the 'Name', 'Ticket' and 'Cabin' columns.
    df.drop('Name', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('PassengerId', axis=1, inplace=True)

    # Convert 'Sex', 'Embarked' and 'Title' columns values into numbers.
    df['Sex'].replace(('male', 'female'), (0, 1), inplace=True)
    df['Embarked'].replace(('S', 'C', 'Q'), (0, 1, 2), inplace=True)
    df['Title'].replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Royalty', 'Officer'), (0, 1, 2, 3, 4, 5, 6, 7), inplace=True)

    # print(df.sample(20))
    # corr = df.corr()
    # corr = corr['Survived'].sort_values(ascending=False)
    # print(corr)
    return df
