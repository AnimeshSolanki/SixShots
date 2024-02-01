import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

def result(row):
    return 1 if row['winner'] == row['batting_team'] else 0

#this df contains total score made by teams playing in 1st innings of all matches
total_score = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score = total_score[total_score['inning']==1]

# this df includes information about the total runs scored in the first inning for each match.
matches_df = matches.merge(total_score[['match_id', 'total_runs']], left_on='id', right_on='match_id')

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

matches_df['team1'] = matches_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
matches_df['team2'] = matches_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

matches_df['team1'] = matches_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
matches_df['team2'] = matches_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

matches_df = matches_df[matches_df['team1'].isin(teams)]
matches_df = matches_df[matches_df['team2'].isin(teams)]

matches_df = matches_df[matches_df['dl_applied'] == 0]

matches_df = matches_df[['match_id', 'city', 'winner', 'total_runs']]

# this df contains info about deliveries of 2nd inning of all matches with additional info from match_df
deliveries_df = matches_df.merge(deliveries, on='match_id')
deliveries_df = deliveries_df[deliveries_df['inning'] == 2]

deliveries_df['curr_score'] = deliveries_df.groupby('match_id')['total_runs_y'].cumsum()
deliveries_df['runs_left'] = deliveries_df['total_runs_x'] - deliveries_df['curr_score']
deliveries_df['balls_left'] = 126 - (deliveries_df['over'] * 6 + deliveries_df['ball'])

deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].fillna("0")
deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].astype('int')
wickets = deliveries_df.groupby('match_id')['player_dismissed'].cumsum().values
deliveries_df['wickets_left'] = 10-wickets

deliveries_df['curr_rr'] = (deliveries_df['curr_score'] * 6)/ (120 - deliveries_df['balls_left'])
deliveries_df['req_rr'] = (deliveries_df['runs_left'] * 6)/deliveries_df['balls_left']

deliveries_df['result'] = deliveries_df.apply(result, axis=1)

final_df = deliveries_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','curr_rr','req_rr','result']]

final_df = final_df.sample(final_df.shape[0])
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

X = final_df.iloc[:, :-1]
Y = final_df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

trf = ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])],remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print("Accuracy Score: ", accuracy_score(Y_test,y_pred))

pickle.dump(pipe,open('pipe.pkl','wb'))