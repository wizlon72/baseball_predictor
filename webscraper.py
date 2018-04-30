from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests

def get_latest_games():
    page_link = 'http://mlb.mlb.com/news/probable_pitchers/index.jsp'
    page_response = requests.get(page_link, timeout=5)
    page_content = BeautifulSoup(page_response.content, "html.parser")
    games = page_content.find_all(class_='module')
    homepitchers = page_content.find_all(class_='pitcher last')
    times = page_content.find_all(class_='pitcher first')
    todaysgamestwo = pd.DataFrame(columns=['Home_Team','Visiting_Team','V_Starter_Name'])
    h_starting_pitchers = pd.DataFrame(columns=['H_Starter_Name'])
    start_times = pd.DataFrame(columns=['Day_Night_D','Day_Night_N'])
    at_regex = " @ "
    comma_regex = ","
    t_regex = "T"
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
    for div in times:
        longtime = div.attrs['local_time']
        time = longtime.split(t_regex)[1]
        if time < '17:00:00':
            start_times=start_times.append(pd.DataFrame([['1','0']], columns=['Day_Night_D','Day_Night_N']),ignore_index=True)
        else:
            start_times=start_times.append(pd.DataFrame([['0','1']], columns=['Day_Night_D','Day_Night_N']),ignore_index=True)
    todaysgamestwo = todaysgamestwo.join(h_starting_pitchers)
    todaysgamestwo = todaysgamestwo.join(start_times)
    todaysgamestwo.to_csv('testingdata.csv')

if __name__ == '__main__':
    get_latest_games()
