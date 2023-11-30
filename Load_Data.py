from bs4 import BeautifulSoup
import urllib.request, urllib.parse, urllib.error
import sqlite3
import time

# Creating an SQL data base
con = sqlite3.connect('HockeyDataBase.sqlite')
curr = con.cursor()
# Creating a new database
curr.executescript('''
DROP TABLE IF EXISTS Performance;

CREATE TABLE Performance (
Team_Name TEXT,
Year_Played INTEGER,
Wins INTEGER,
Losses INTEGER,
OT_Losses INTEGER,
Goals_For INTEGER,
Goals_Against INTEGER
);
''')


# Orginal URL
base_url = 'https://www.scrapethissite.com/pages/forms/?page_num='
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
}

current_page = 1
while current_page <= 6:
    page_url = f'{base_url}{current_page}&per_page=100'
    print(page_url)
    req = urllib.request.Request(page_url, headers=header)

    url = urllib.request.urlopen(req)
    # Reading the data
    data = url.read()
    # Parsing HTML Content
    soup = BeautifulSoup(data, 'html.parser')
    team_info = soup.find_all('tr', class_='team')
    for i in team_info:
        td = i.find_all('td')
        # Extracting text content from team_info
        name = td[0].get_text(strip=True)
        year = td[1].get_text(strip=True)
        wins = td[2].get_text(strip=True)
        losses = td[3].get_text(strip=True)
        ot = td[4].get_text(strip=True)
        goals_for = td[6].get_text(strip=True)
        goals_against = td[7].get_text(strip=True)

        # Checking if names been collected
        print(name, year, wins, losses,ot, goals_against, goals_for)
        # Adding Data To Data Base
        curr.execute(
            '''INSERT INTO Performance(Team_Name, Year_Played, Wins, Losses,OT_Losses, Goals_For, Goals_Against) VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (name, year, wins, losses,ot, goals_for, goals_against))

        con.commit()
    current_page += 1
    time.sleep(1)
























