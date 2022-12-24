#importing libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

#my CSV file
df = pd.read_csv('~/downloads/league_player_data.csv')

#defining "variables" (dataframes.. remember df = dataframes)
X = df[['playerOneLevel', 'playerTwoLevel', 'playerThreeLevel', 'playerFourLevel', 'playerFiveLevel', 'properRunesSums']]
y = df['WinLose']

model = LinearRegression()

model.fit(X.values, y)

#defining values
playerOneLevel = 10
playerTwoLevel = 100
playerThreeLevel = 400  
playerFourLevel = 220  
playerFiveLevel = 210
properRunesSums = 1
prediction = model.predict([[playerOneLevel,playerTwoLevel,playerThreeLevel,playerFourLevel,playerFiveLevel,properRunesSums]])

#maxing and minning at 0 and 1
finalPrediction = min(prediction, 1)
finalPrediction = max(0,finalPrediction)

print(finalPrediction)
