import webscraper
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

def addWinner(df):
    visitors = df.Visiting_Team_Score > df.Home_Team_Score
    home = df.Home_Team_Score > df.Visiting_Team_Score
    column_name = 'Winner'
    df.loc[visitors, column_name] = df['Visiting_Team']
    df.loc[home, column_name] = df['Home_Team']
    return df

def homeWinner(df):
    visitors = df.Visiting_Team_Score > df.Home_Team_Score
    column_name = 'homeTeamWin'
    df['homeTeamWin'] = 1
    df.loc[visitors,column_name] = 0
    return df

def pitcherERA(df1, df2):
    df2 = df2[['ERA']]
    visitors = df2.rename(columns={'ERA':'V_S_ERA'})
    home = df2.rename(columns={'ERA':'H_S_ERA'})
    df1 = df1.join(visitors, on='V_Starter_Name')
    df1 = df1.join(home, on='H_Starter_Name')
    df1['ERA_diff']=df1['H_S_ERA']-df1['V_S_ERA']
    return df1

def trimForRegression(df1):
# Just building on ERA diff
#    dftest = df1[['homeTeamWin','ERA_diff']]
#    [[ 98 127]
#    [ 79 213]]
# ERA + Day Night
    dftest = df1[['homeTeamWin','ERA_diff','Day_Night']]
    dftest = pd.get_dummies(dftest, columns =['Day_Night'])
# Building on ERA and Team
#    dftest = df1[['homeTeamWin','ERA_diff','Visiting_Team','Home_Team','Day_Night']]
#    dftest = pd.get_dummies(dftest, columns =['Visiting_Team','Home_Team', 'Day_Night'])
#    [[ 95 130]
#    [100 192]]
# Adding Day/Night
#[[ 95 130]
# [ 97 195]]
# Clean up NA's
    dftest.dropna(axis=0, how='any',inplace = True)
    return dftest

def testToday(model,eradata):
    webscraper.get_latest_games()
    testset = pd.read_csv('testingdata.csv')
    testset = pitcherERA(testset,eradata)
    testset.dropna(axis=0, how='any',inplace = True)
    testset.reset_index(drop=True,inplace=True)
    concat = testset[['ERA_diff','Day_Night']]
    concat = pd.get_dummies(concat, columns =['Day_Night'])
    results = model.predict_proba(concat)
    resultsdf = pd.DataFrame(results.reshape(results.shape[0],2))
    resultsdf.rename(columns={0:"prob_VisTeamWin",1:"prob_HomTeamWin"},inplace=True)
    testset = testset.join(resultsdf)
    print('Predictions for games today')
    print(testset)

olddata = pd.read_csv('GL2017.csv' )
pitcherdata = pd.read_csv('pitchers.csv', index_col='Name')
newdf = addWinner(olddata)
newdf = homeWinner(newdf)
newdf = pitcherERA(newdf, pitcherdata)
data = trimForRegression(newdf)
X = data.iloc[:,1:]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test,y_pred)
print ('Confusion matrix from train/test on historical data:')
print(confusion_matrix)
print ('Coefficients for independent variables')
print (classifier.coef_)
testToday(classifier,pitcherdata)
