import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandasgui
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import mlflow


def todo1():
    df = pd.read_csv('phpMYEkMl.csv')
    df.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)

    """ Cleaning missing data """
    # zamiana ? na NaN
    df[df == '?'] = np.NaN
    # clean numerical values and categorical
    df['age'] = pd.to_numeric(df['age'])
    df['fare'] = pd.to_numeric(df['fare'])
    # lets analyze embarked first
    # print(df[df['embarked'].isnull()])
    # we fill it by searching info about the people in google
    df['embarked'] = df['embarked'].fillna('S')
    # lets analyze fare now, 1 missing data
    # print(df[df['fare'].isnull()])
    # Median Fare value of a male with a third class ticket and no family is a logical choice to fill the missing value
    median_fare_miss = df.groupby(['pclass', 'parch', 'sibsp'])['fare'].median()[3][0][0]
    df['fare'] = df['fare'].fillna(median_fare_miss)
    # lets analyze age now, a lot of missing data
    # age is correlated with pclass and survived
    df['age'] = df.groupby(['sex', 'pclass'])['age'].apply(lambda x: x.fillna(x.median()))
    # cabin is mostly missing so drop it
    df.drop('cabin', inplace=True, axis=1)

    """ Feature engineering """

    # extracting meanningful features from actual and groupping
    # lets go through all variables and encode cat variables and extract meaningnful features
    # so we can make a new variable family_size from parch and sibsp and group it
    df['familysize'] = df['sibsp'] + df['parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
                  9: 'Large', 10: 'Large', 11: 'Large'}
    df['family_size_grouped'] = df['familysize'].map(family_map)
    df.drop(['familysize', 'sibsp', 'parch'], axis=1, inplace=True)
    # extract titles from name
    df['title'] = df['name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['is_married'] = 0
    df['is_married'].loc[df['title'] == 'Mrs'] = 1
    df.drop('name', inplace=True, axis=1)
    # lets take a look on continous data
    # lets analyze age and group it with qcut based on normal dist
    df['age'] = pd.qcut(df['age'], 10)
    # lets analyze fare and group it with qcut based on normal dist
    df['fare'] = pd.qcut(df['fare'], 13)
    # lets take a look on nominal data
    df['ticket_freq'] = df.groupby('ticket')['ticket'].transform('count')
    df.drop('ticket', axis=1, inplace=True)
    # end
    # encoding
    non_numeric_features = ['embarked', 'sex', 'title', 'family_size_grouped', 'age', 'fare']
    enc = LabelEncoder()
    for feature in non_numeric_features:
        df[feature] = enc.fit_transform(df[feature])

    cat_features = ['pclass', 'sex', 'embarked', 'title', 'family_size_grouped']
    encoded_features = []
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

    df = pd.concat([df, *encoded_features], axis=1)
    df.drop(cat_features, axis=1, inplace=True)
    X = df.drop('survived', axis=1)
    y = df['survived']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y,
                                                        test_size=0.1)
    #
    parameters = {'n_estimators': [100, 200, 500],
                  'criterion': ['gini', 'entropy']}
    mlflow.sklearn.autolog()
    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10)
    clf.fit(x_train, y_train)

    pvt = pd.pivot_table(
        pd.DataFrame(clf.cv_results_),
        values='mean_test_score',
        index='param_criterion',
        columns='param_n_estimators'
    )

    ax = sns.heatmap(pvt)
    plt.show()

    # SPRAWDZENIE DYSTRYBUCJI W NUMERYCZNYCH WARTOSCIACH
    # print(df.describe())
    # SPRAWDZENIE DYSTRUBCJI W KATEGORYCZNYCH WARTOSCIACH
    # print(df.describe(include=['O']))
    # NALEZY SPRAWDZIC KORELACJE KAZDEJ WARTOSCI Z PREDYKTOWANA CZYLI SURVIVED
    # ZASTANOWIENIE SIE KTORA CECHA MA DUZA ILOSC DUPLIKATOW
    # CZYSZCZEDZENIE DANYCH


todo1()
