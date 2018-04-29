import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
#import seaborn as sns
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)
#import matplotlib.pyplot as plt
#from sklearn import linear model
def addWinner(df):
    visitors = df.Visiting_Team_Score > df.Home_Team_Score
    home = df.Home_Team_Score > df.Visiting_Team_Score
    column_name = 'Winner'
    df.loc[visitors, column_name] = df['Visiting_Team']
    df.loc[home, column_name] = df['Home_Team']
    #df = df.assign(Winner=)
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
    #df2.set_index(['Name'])
    #print (df2.head(4))
    #print (df3.head(4))
    df1 = df1.join(visitors, on='V_Starter_Name')
    df1 = df1.join(home, on='H_Starter_Name')
    df1['ERA_diff']=df1['H_S_ERA']-df1['V_S_ERA']
    return df1

def trimForRegression(df1):
# Including after the fact data
    #dftest = df1[['homeTeamWin','ERA_diff','Visiting_Team_Score']]
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
    testset = pd.read_csv('testingdata.csv')
#    visitors = eradata.rename(columns={'ERA':'V_S_ERA'})
#    home = eradata.rename(columns={'ERA':'H_S_ERA'})
#    testset = testset.join(visitors, on='V_Starter_Name')
#    testset = testset.join(home, on='H_Starter_Name')
#    testset['ERA_diff']=testset['H_S_ERA']-testset['V_S_ERA']
    testset = pitcherERA(testset,eradata)
    testset.dropna(axis=0, how='any',inplace = True)
    concat = testset[['ERA_diff','Day_Night']]
    concat = pd.get_dummies(concat, columns =['Day_Night'])
    results = model.predict_proba(concat)
    resultsdf = pd.DataFrame(results.reshape(3,2))
    resultsdf.rename(columns={0:"prob_VisTeamWin"},inplace=True)
    resultsdf.rename(columns={1:"prob_HomTeamWin"},inplace=True)
    testset = testset.join(resultsdf)
#    return results
    print(testset)
olddata = pd.read_csv('GL2017.csv' )
pitcherdata = pd.read_csv('pitchers.csv', index_col='Name')
#print (olddata)
newdf = addWinner(olddata)
newdf = homeWinner(newdf)
newdf = pitcherERA(newdf, pitcherdata)
#newdf = newdf['V_Starter','Winner','homeTeamWin','Name','ERA']
#print olddata.head(3)
#print "is this working"
# Plot showing distributions of 0's and 1's
    #sns.countplot(x='homeTeamWin',data=newdf, palette='hls')
    #plt.show()
data = trimForRegression(newdf)
X = data.iloc[:,1:]
#print (X)
y = data.iloc[:,0]
#print (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#print(X_train.shape)
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print (classifier.coef_)
testToday(classifier,pitcherdata)
