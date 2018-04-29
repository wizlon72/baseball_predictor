from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests

page_link = 'http://mlb.mlb.com/news/probable_pitchers/index.jsp'
page_response = requests.get(page_link, timeout=5)
page_content = BeautifulSoup(page_response.content, "html.parser")
games = page_content.find_all(class_='module')
homepitchers = page_content.find_all(class_='pitcher last')
visitingpitchers = page_content.find_all(class_='pitcher first')
todaysgames =pd.DataFrame(columns=['Home_Team','Visiting_Team','H_Starter_Name','V_Starter_Name','Day_Night'])
todaysgamestwo = pd.DataFrame(columns=['Home_Team','Visiting_Team','V_Starter_Name'])
h_starting_pitchers = pd.DataFrame(columns=['H_Starter_Name'])
at_regex = " @ "
comma_regex = ","
for tag in games:
    teams = tag.h4.text
    teams_split = teams.split(at_regex)
    Visiting_Team = teams_split[0]
    Home_Team = teams_split[1]
    V_Starter_Name = tag.h5.text.split(comma_regex)[0]
    todaysgamestwo = todaysgamestwo.append(pd.DataFrame([[Home_Team,Visiting_Team,V_Starter_Name]],columns=['Home_Team','Visiting_Team','V_Starter_Name']),ignore_index=True)
for tag in homepitchers:
    H_Starter_Name = tag.h5.text.split(comma_regex)[0]
    h_starting_pitchers = h_starting_pitchers.append(pd.DataFrame([[H_Starter_Name]], columns=['H_Starter_Name']),ignore_index=True)
#    print H_Starter_Name
#    print(tag)
#    H_Starter_Name = homemodule.h5.text.split(comma_regex)[0]
#    print(Visiting_Team)
#    print(Home_Team)
#    print(V_Starter_Name)
todaysgamestwo = todaysgamestwo.join(h_starting_pitchers)
print(todaysgamestwo)
