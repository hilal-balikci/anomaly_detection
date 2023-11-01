import numpy as np
import pandas as pd 

# Extra Libs
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from bokeh.models import HoverTool
from IPython.display import HTML, display
import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest

df = pd.read_csv('ambient_temperature_system_failure.csv' , sep =";")

print(df.shape) 

print(df.info())

print(df.describe())

print(df['value'].mean())

#check the duplicates in the dataset
print('duplicated rows: ', df.duplicated().sum()) #duplicated rows:  0

# change the type of timestamp column for plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])
# change fahrenheit to °C (temperature mean= 71 -> fahrenheit)
df['value'] = (df['value'] - 32) * 5/9
# plot the data
df.plot(x='timestamp', y='value')



df['x1']= df['value'].shift(+1)
df['x2'] = df['value'].shift(+2)

df.dropna(inplace = True)

x1,x2,y = df['x1'],df['x2'], df['value']
x1,x2, y = np.array(x1), np.array(x2), np.array(y)
x1, x2,  y = x1.reshape(-1,1), x2.reshape(-1,1),  y.reshape(-1,1)
joined_x = np.concatenate((x1, x2), axis = 1)

print(joined_x.shape)

#split the dataset into two as training and testing
#0.80 for training, 0.20 for testing

split_time = 5812
X_train = joined_x[:split_time]
y_train = y[: split_time]
x_test = joined_x[split_time:]
y_test = y[ split_time:]

print(X_train)

regressor = LinearRegression()
regressor.fit( X_train, y_train)
y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)
print(y_pred.shape) #(1453, 1)

pd.DataFrame(y_pred).plot(xlabel = 'time steps', ylabel = 'Temp. °f', figsize = (8,4), grid = True, title = 'Linear Regression Prediction ', )

pd.DataFrame(y_test).plot(xlabel = 'time steps', ylabel = 'Temp. °f', figsize = (8,4), grid = True, title = 'Real Values ',)

plt.figure(figsize=(8, 6),)
plt.plot(y_pred, label = 'linear Regression Prediction')
plt.plot(y_test, label = 'Actual Signal')
plt.legend(loc = "upper left")
plt.show()

df_copy = df.drop(['timestamp'], axis=1)

print(df_copy)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, np.ravel(y_train))
tree_pred = regr.predict(x_test)

pd.DataFrame(tree_pred).plot(xlabel = 'time steps', ylabel = 'Temp. °f', figsize = (8,4), grid = True, title = ' Random forest regression prediction ', )

plt.figure(figsize=(8, 6))
plt.plot(tree_pred, label = 'random forest regression prediction')
plt.plot(y_test, label = 'Actual signal')
plt.legend(loc = "upper left")
plt.show()

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(regr, X_train, np.ravel(y_train), scoring='r2', cv=folds)
print(scores) 

kmeans = KMeans(n_clusters = 1).fit(df_copy)
print(kmeans)
print((mean_squared_error(tree_pred, y_test)))

center = kmeans.cluster_centers_
print(center)

distance = sqrt((df_copy - center)**2)
print(distance)


#variable to determine how many farthest points we will label as anomalies
number_of_points = 25

order_index = argsort(distance, axis = 0)
indexes = order_index[-number_of_points:].values
index_values = []

for i in indexes:
    index_values.append(i[0])
    
values = [df_copy.iloc[i] for i in index_values]

plt.plot(df_copy)
plt.scatter(indexes, values, color='r')
plt.show()

#Feature Engineering 

# the hours and if it's night or day (7:00-22:00)
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

# the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
# An estimation of anomly population of the dataset (necessary for several algorithm)
outliers_fraction = 0.01

# time with int to plot easily
df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

# creation of 4 distinct categories that seem useful (week end/day week & night/day)
df['categories'] = df['WeekDay']*2 + df['daylight']

a = df.loc[df['categories'] == 0, 'value']
b = df.loc[df['categories'] == 1, 'value']
c = df.loc[df['categories'] == 2, 'value']
d = df.loc[df['categories'] == 3, 'value']

fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(a)
b_heights, b_bins = np.histogram(b, bins=a_bins)
c_heights, c_bins = np.histogram(c, bins=a_bins)
d_heights, d_bins = np.histogram(d, bins=a_bins)

width = (a_bins[1] - a_bins[0])/6

ax.bar(a_bins[:-1], a_heights*100/a.count(), width=width, facecolor='blue', label='WeekEndNight')
ax.bar(b_bins[:-1]+width, (b_heights*100/b.count()), width=width, facecolor='green', label ='WeekEndLight')
ax.bar(c_bins[:-1]+width*2, (c_heights*100/c.count()), width=width, facecolor='red', label ='WeekDayNight')
ax.bar(d_bins[:-1]+width*3, (d_heights*100/d.count()), width=width, facecolor='black', label ='WeekDayLight')

plt.legend()
plt.show()


#Isolation Forest
# Take useful feature and standardize them 
data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# train isolation forest 
model =  IsolationForest(contamination = outliers_fraction)
model.fit(data)
# add the data to the main  
df['anomaly25'] = pd.Series(model.predict(data))
df['anomaly25'] = df['anomaly25'].map( {1: 0, -1: 1} )
print(df['anomaly25'].value_counts())


# visualisation of anomaly throughout time (viz 1)
fig, ax = plt.subplots()

a = df.loc[df['anomaly25'] == 1, ['time_epoch', 'value']] #anomaly

ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'],a['value'], color='red')
plt.show()

def detect_IQR(df,feature):
    lower_half = df[feature].quantile(q=0.25) #Q1
    upper_half = df[feature].quantile(q=0.75) #Q3
    IQR = upper_half - lower_half #define the euqation as IQR = Q3-Q1
    lower_bond_point = lower_half - 3*IQR
    upper_bond_point = upper_half + 3*IQR
    
    return lower_bond_point,upper_bond_point

lower_bond_point,upper_bond_point = detect_IQR(df,"value")
print(lower_bond_point,upper_bond_point)

IQR_result_df=pd.DataFrame()
IQR_result_df['timestamp']=df['timestamp']
IQR_result_df['value'] = df['value']

#Inliers are labeled 1, while outliers are labeled -1.

IQR_result_df.loc[(IQR_result_df["value"]<lower_bond_point) | (IQR_result_df["value"]>upper_bond_point),"anomaly"] = 1

import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=IQR_result_df['timestamp'], y=IQR_result_df['value'],
                    mode='lines',
                    name='lines'))

a=IQR_result_df[IQR_result_df['anomaly']==1]

fig.add_trace(go.Scatter(x=a.timestamp, y=a.value,
                    mode='markers',
                    name='markers'))

fig.update_layout(title='Anomaly detection using IQR')
fig.show("notebook")
    





