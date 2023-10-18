# f1-analysis-2023
**Andreas Nikolaidis** 

_October 2023_

- [Introduction](#introduction)
- [Visualization & Exploratory Analysis](#Exploratory)
- [Redbull](#Redbull)
- [Classification](#Classification)

## [Introduction](#introduction)
In this notebook, we will look at the historical performance of drivers and constructors and compare the years.

Start by importing all the necessary packages into Python:
```python
import numpy as np
from numpy import outer
import pandas as pd
from pandas import Series, DataFrame
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go

sns.set_style('whitegrid')
%matplotlib inline
from matplotlib.gridspec import GridSpec

# Import for Linear Regression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
```

Import files:
```python
import os
for dirname, _, filenames in os.walk('/Users/~/csv_files'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
Combine:
```python
import os
import pandas as pd

folder_path = "/Users/~/csv_files"
dataset = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Specify the encoding as 'latin-1' to handle non-UTF-8 encoded files
            dataset[file_name.replace(".csv", "_df")] = pd.read_csv(file_path, encoding='latin-1')
        except UnicodeDecodeError as e:
            print(f"Error reading {file_name}: {str(e)}")

print(f'List of dataframes available: {dataset.keys()}')
```
List of dataframes available: dict_keys(['circuits_df', 'status_df', 'lap_times_df', 'sprint_results_df', 'drivers_df', 'races_df', 'constructors_df', 'constructor_standings_df', 'qualifying_df', 'driver_standings_df', 'constructor_results_df', 'pit_stops_df', 'seasons_df', 'results_df'])

```python
plt.style.use('dark_background')
colormap = plt.cm.get_cmap('coolwarm')
```

## [Visualization & Exploratory Analysis](#Exploratory)

```python
merged_data = pd.merge(constructor_standings, constructors, on='constructorId')
constructor_points = merged_data.groupby('name')['points'].sum()
top_constructors = constructor_points.sort_values(ascending=False).head(10)
```
```python
# Create the figure with a dark background
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor('black')

# Plot your data with a light blue color
top_constructors.plot(kind='bar', color='lightblue')

# Customize the text labels and title to be white
plt.xlabel('Constructor', color='white')
plt.ylabel('Total Points', color='white')
plt.title('Top 10 Most Successful Constructors in Formula 1 History', color='white')

# Customize the tick labels
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')

plt.show()
```
![db84b0a7-d708-45fb-bbef-ff1a17044c45](https://github.com/atnikola/f1-analysis-2023/assets/38530617/e83fb3ef-bf8e-48b0-9964-95e1d2caa2ff)

```python
# Set a dark background style
plt.style.use("dark_background")

# Create the figure with a dark background
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor('black')

# Plot your data with white text labels
plt.bar(yearly_race_counts['year'].astype("str"), yearly_race_counts['raceId'], color='lightblue')
plt.xlabel('Year', color='white')
plt.ylabel('Number of Races', color='white')
plt.title('Number of Formula 1 Races by Year', color='white')

# Customize the tick labels
plt.xticks(color='white')
plt.yticks(color='white')

plt.show()
```
```python
merge=constructors.merge(results,on = 'constructorId',how = 'left')
the_best = merge[['name','points','raceId']]
the_best = the_best.groupby('name')['raceId'].nunique(10).sort_values(ascending = False).reset_index(name = 'races')
the_best = the_best[the_best['races'] >= 100]
the_best.head()
```
<img width="141" alt="Screenshot 2023-10-18 at 14 27 19" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/e6578109-210d-474f-9f1e-5f451b7c8b88">

```python
total_points = merge.groupby('name')['points'].sum().reset_index()
races_participated = merge.groupby('name')['raceId'].nunique().reset_index()
result = pd.merge(total_points, races_participated, on='name')
result['points_per_race'] = result['points'] / result['raceId']
result = result.sort_values(by='points_per_race', ascending=False)
result.head(10)
```
<img width="375" alt="Screenshot 2023-10-18 at 14 30 40" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/16baf372-e13f-49bb-b0eb-88228feaf3a9">

```python
fig = gr.Figure(data=[gr.Bar(x=result['name'], y=result['points_per_race'])])
fig.update_layout(
    title_text="Constructor's Points Per Race",
    xaxis_title='Constructor',
    yaxis_title='Points Per Race',
    xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2, tickfont=dict(size=12)),
    yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2, tickfont=dict(size=12)),
    plot_bgcolor='black',  # Set the plot background color to black
    paper_bgcolor='black',  # Set the paper (border) background color to black
    font=dict(color='white'),  # Set the font color to white
)
fig.update_traces(textfont_size=29,
                  marker=dict(line=dict(width=2)))
fig.show()
```
<img width="1584" alt="Screenshot 2023-10-18 at 14 31 28" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/7e389f1b-4eb3-42ab-8a33-ea7744151c8c">

(Lots of failed teams :/)

Additionally the average ages of drivers have consistently dropped - let's take a look at the most data for 2023
```python
current_year = 2023  
drivers['age'] = current_year - drivers['dob'].apply(lambda x: int(x.split('-')[0]))
age_groups = ['18-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51+']
drivers['age_group'] = pd.cut(drivers['age'], bins=[18, 25, 30, 35, 40, 45, 50, float('inf')], labels=age_groups)
age_group_counts = drivers['age_group'].value_counts()


# Set a dark background style
plt.style.use("dark_background")

fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('black')

current_year = 2023
drivers['age'] = current_year - drivers['dob'].apply(lambda x: int(x.split('-')[0]))
age_groups = ['18-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51+']
drivers['age_group'] = pd.cut(drivers['age'], bins=[18, 25, 30, 35, 40, 45, 50, float('inf')], labels=age_groups)
age_group_counts = drivers['age_group'].value_counts()

colors = plt.cm.Blues(np.linspace(0.8, 0.2, len(age_group_counts)))

plt.pie(age_group_counts, labels=age_group_counts.index, autopct="%1.1f%%", startangle=170, colors=colors)
plt.title('Distribution of Driver Ages in Formula 1', color='white')
plt.axis('equal')
plt.show()

```
![abd00c91-362a-4edc-be64-69d731ed6a42](https://github.com/atnikola/f1-analysis-2023/assets/38530617/27bab1c4-f29d-4657-a4c2-3c1fdb1dc28f)

```python
# Set a dark background style
plt.style.use("dark_background")

# Create the figure with a dark background
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor('black')

# Plot your data with white text labels
win_percentages.sort_values().plot(kind='bar', color='lightpink')
plt.xlabel('Circuit', color='white')
plt.ylabel('Win Percentage (%)', color='white')
plt.title('Circuit Win Percentages in Formula 1', color='white')

# Customize the tick labels
plt.xticks(rotation=90, color='white')

#Remove gridlines
plt.grid(False)

plt.tight_layout()
plt.show()
```
![639ecb95-cd67-4008-a46b-16b48294b388](https://github.com/atnikola/f1-analysis-2023/assets/38530617/6126d6cc-6726-43eb-8727-7ad4a706de1e)

```python
# Set a dark background style
plt.style.use("dark_background")

# Create the figure with a dark background
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor('black')

# Plot your data with white text labels and no grid lines
plt.fill_between(pitstops_count['name'], pitstops_count['raceId'], color='orange', alpha=0.7)
plt.xlabel('Circuit', color='white')
plt.ylabel('Number of Pit Stops', color='white')
plt.title('Number of Pit Stops at Different Circuits in Formula 1', color='white')

# Customize the tick labels
plt.xticks(rotation=90, color='white')

# Remove grid lines
plt.grid(False)

plt.tight_layout()
plt.show()
```
![1a9c66e2-6110-4e96-97f2-2ac938d44f96](https://github.com/atnikola/f1-analysis-2023/assets/38530617/957c8cd3-030a-4df2-90d9-f94623e5de01)

Most GP F1 Winners
```python
#top 10 drivers plot

sb.barplot(data=top10Drivers,y='driver',x='positionOrder',color='blue',alpha=0.8,linewidth=.8)
plt.title('Top 10 Most GP Winners in F1')
plt.grid(False)
plt.ylabel('Driver Name')
plt.xlabel('Number of GP wins')
```
![ab64ca4b-0c35-4c69-b7c6-47b29ab7a557](https://github.com/atnikola/f1-analysis-2023/assets/38530617/4a5c2144-11b4-4ed4-bcd3-ea19510f34a5)



Nationality:
```python
plt.style.use("dark_background")

fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor('black')

nationality_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Nationality', color='white')
plt.ylabel('Number of Drivers', color='white')
plt.title('Number of Drivers in Formula 1 by Nationality', color='white')

plt.xticks(rotation=90, color='white')

# Remove grid lines
plt.grid(False)

plt.tight_layout()
plt.show()
```

Worst Tracks based on overtakes:
```python
merged_data = pd.merge(circuits, races, on='circuitId')

circuit_usage_counts = merged_data['circuitId'].value_counts()
top_circuits = circuit_usage_counts.head(10)

merged_data = pd.merge(races, results, on='raceId')
overtaking_rate = merged_data.groupby('circuitId')['positionOrder'].apply(lambda x: (x != 1).sum() / len(x)).reset_index()
overtaking_rate.columns = ['circuitId', 'overtake_rate']
overtaking_rate = pd.merge(overtaking_rate, circuits, on='circuitId')
worst_tracks = overtaking_rate.sort_values(by='overtake_rate', ascending=True)
```

```python
plt.figure(figsize=(12, 6))
plt.bar(worst_tracks['name'], worst_tracks['overtake_rate'], color='lightblue')
plt.xlabel('Circuit')
plt.ylabel('Overtaking Rate')
plt.title('Worst Tracks Based on Overtaking Action')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.grid(False)
plt.show()
```
![5bb323d4-8903-4b75-9df4-b54b9a46a521](https://github.com/atnikola/f1-analysis-2023/assets/38530617/33a1ad1d-be7e-4df1-a260-7093d87d57fe)

Worst Tracks based on average speed:
```python
merged_data = pd.merge(races, laptimes, on='raceId')
merged_data['average_speed'] = merged_data['milliseconds'] / merged_data['milliseconds'].max() 
average_speed_by_track = merged_data.groupby('circuitId')['average_speed'].mean().reset_index()
average_speed_by_track = pd.merge(average_speed_by_track, circuits, left_on='circuitId', right_on='circuitId')
worst_tracks = average_speed_by_track.sort_values(by='average_speed')
```

```python
plt.figure(figsize=(16, 6))
plt.bar(worst_tracks['name'], worst_tracks['average_speed'], color='purple')
plt.xlabel('Circuit')
plt.ylabel('Average Race Speed')
plt.title('Worst Tracks Based on Average Race Speed')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.grid(False)
plt.show()
```
![22f79fb6-66b9-460e-a414-6babcdd0a958](https://github.com/atnikola/f1-analysis-2023/assets/38530617/8cae202f-6cef-4440-a725-0a55698fb228)

Average Laptimes over the years:
```python
import seaborn as sns

merged_data = pd.merge(races, laptimes, on='raceId')
merged_data = pd.merge(merged_data, drivers, on='driverId')
average_lap_time_by_race = merged_data.groupby(['year', 'raceId'])['milliseconds'].mean().reset_index()

plt.figure(figsize=(16, 6))
sns.lineplot(x='year', y='milliseconds', data=average_lap_time_by_race)
plt.xlabel('Year')
plt.ylabel('Average Lap Time (milliseconds)')
plt.title('Trend of Average Lap Times in Races Over the Years')
plt.grid(False)
plt.show()
```
![a85506de-d0d1-42ba-a50b-9fe31b0803df](https://github.com/atnikola/f1-analysis-2023/assets/38530617/3688a6e1-f6de-40cc-873c-217742ad4f90)


Next I want to create a simple dashboard that allows us to select a race and have it display the fastest lap times every year next to a box plot of the full range. 
```python
from ipywidgets import interact, widgets
import matplotlib.gridspec as gridspec
import numpy as np

circuits = fastest_laptime_years_circuits.keys()

def plot_graph(circuit):
    circuit_data = fastest_laptime_years_circuits[circuit]
    years_ = list(circuit_data.keys())
    fastest_lap_ = list(circuit_data.values())
    
    colors = np.linspace(0, 1, len(years_))

    # Create a grid with 1 row and 2 columns for both plots
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Line plot
    ax1 = plt.subplot(gs[0])
    ax1.set_xticks(years_)
    ax1.set_xticklabels(years_, rotation=45, ha='right')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Fastest Lap Timing')
    ax1.set_title(f'{circuit} Yearly Fastest Lap Timing')

    for i in range(len(years_) - 1):
        ax1.plot([years_[i], years_[i + 1]], [fastest_lap_[i], fastest_lap_[i + 1]], color=colormap(colors[i % len(colors)]),
                 label='Red Bull Racing')
    # Box plot
    ax2 = plt.subplot(gs[1])
    ax2.boxplot(fastest_lap_, vert=True)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Fastest Lap Timing')
    ax2.set_title(f'{circuit} Fastest Lap Timing Box Plot')

    plt.show()
    
    range_of_years = (min(years_), max(years_))
    faster_or_slower = fastest_lap_[-1] - fastest_lap_[0]
    faster_or_slower_word = ""
    if faster_or_slower > 0 :
        faster_or_slower_word = "SLOWER"
    elif faster_or_slower < 0 :
        faster_or_slower_word = "FASTER"
    else:
        faster_or_slower_word = "UNCHANGED"
    faster_or_slower = abs(faster_or_slower)
    print(f'Short Summary : ')
    print(f'{circuit} raced in this circuit from {range_of_years[0]} to {range_of_years[1]}')
    print(f'They became {faster_or_slower_word} by {faster_or_slower} seconds')

# Create the interactive dropdown
interact(plot_graph, circuit=widgets.Dropdown(options=circuits))
print()
```
![Oct-18-2023 13-29-42](https://github.com/atnikola/f1-analysis-2023/assets/38530617/402b39f1-4b94-42e3-a9e8-0c1a8d6d2456)

Extra Visualization of Races using Folium Maps:
```python
circuit_df = pd.read_csv('/Users/~/circuits.csv')
circuit_df.head()
```
<img width="1011" alt="Screenshot 2023-10-18 at 15 14 03" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/4f364038-b873-408f-9348-9438afd13341">

```python
# ploting the f1 track using lat and lng in worldmap

#pip install folium
import folium
coordinates=[]
for lat,lng in zip(circuit_df['lat'],circuit_df['lng']):
    coordinates.append([lat,lng])
maps = folium.Map(zoom_start=150,tiles='Stamen Watercolor')  #map_types (Stamen Terrain, Stamen Toner, Mapbox Bright, cartodbpositron)
for i,j in zip(coordinates,circuit_df.name):
    marker = folium.Marker(
        location=i,
        icon=folium.Icon(icon="car",color='cadetblue',prefix='fa'),
        popup="<strong>{0}</strong>".format(j))  #strong is used to bold the font (optional)
    marker.add_to(maps)
maps
```
<img width="1024" alt="Screenshot 2023-10-18 at 15 05 12" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/8c2a3daf-24bd-4bb0-bae6-157a17d3101e">



## [Redbull](#Redbull)

Since Redbull has been dominating for the last several years, let's take a look at their historical performance and entry to Formula One:
```python
# year of RedBull entry to F1:
constructorId = 9
filtered_results_df = dataset["results_df"][dataset["results_df"]["constructorId"] == constructorId]
# print(filtered_results_df.head())
races_participated = list(set(filtered_results_df["raceId"].tolist()))
races_participated.sort()

years_participated = [] #(race_id , year)
for i in races_participated:
    years_participated.append((i,int(dataset["races_df"].loc[dataset["races_df"]["raceId"] == i, "year"].values)))
# print(years_participated)

entry_year = min([x[1] for x in years_participated])
print(f'RedBull started racing from year {entry_year}')
```
RedBull started racing from year 2005

Let's take a look now at their points every year compared to the max performance for that year (i.e. Redbull vs the winning contructor for that year)

```python
points_vs_years = {}  # year : total_points

def getYearOfRace(raceId):
    year = dataset['races_df'].loc[dataset['races_df']['raceId'] == raceId, "year"]
    return year

def getConstructor(constructorId):
    constructor = dataset['constructors_df'].loc[dataset['constructors_df']['constructorId'] == constructorId, "name"]
    return constructor.item()

resultsOfRedbull = dataset['results_df'][(dataset['results_df']['constructorId'] == constructorId)]

for index, row in resultsOfRedbull.iterrows():
    raceId = row['raceId']
    points = int(row['points'])
    year = getYearOfRace(raceId)
    try:
        points_vs_years[int(year)] += points
    except:
        points_vs_years[int(year)] = points

points_vs_years = dict(sorted(points_vs_years.items(), key=lambda x: x[0]))
years = list(points_vs_years.keys())
points = list(points_vs_years.values())

each_constructor_points_years = {}
dataset['constructor_results_df']

for index, row in dataset['constructor_results_df'].iterrows():
    year = int(getYearOfRace(row['raceId']))
    if year >= 2005:
        point = row['points']
        if year in each_constructor_points_years.keys():
            if row['constructorId'] in each_constructor_points_years[year].keys():
                each_constructor_points_years[year][row['constructorId']] += point
            else:
                each_constructor_points_years[year][row['constructorId']] = point
        else:
            each_constructor_points_years[year] = {row['constructorId']: point}

max_points_years = {}
for i in each_constructor_points_years.keys():
    constructor_id = max(each_constructor_points_years[i], key=lambda k: each_constructor_points_years[i][k])
    max_points = max(each_constructor_points_years[i].values())
    max_points_years[i] = [max_points, constructor_id]

max_points_years = dict(sorted(max_points_years.items(), key=lambda x: x[0]))

years = list(max_points_years.keys())
max_points = [x[0] for x in list(max_points_years.values())]
constructor_ids = [x[1] for x in list(max_points_years.values())]
constructors = []
for i in constructor_ids:
    constructors.append(getConstructor(i))
# colormap = plt.cm.get_cmap('RdYlBu')
colors = np.linspace(0, 1, len(years))
fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.bar(years, max_points, color=colormap(np.linspace(0, 1, len(years))), alpha=0.4)

for i, bar in enumerate(bars):
    height = bar.get_height()
    offset = 0.2 * height  # Set the offset as a fraction of the bar height
    ax1.text(bar.get_x() + bar.get_width() / 2, height - offset, constructors[i], ha='center', va='top', rotation='vertical')

ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45, ha='right')

ax1.set_xlabel('Years')
ax1.set_ylabel('Max Points')
ax1.set_title('Red Bull Racing Points Each Year vs Max. Points Awarded')
# colors = [colormap(i / len(points)) for i in range(len(points))]
ax2 = ax1.twinx()
for i in range(len(years) - 1):
    ax2.plot([years[i], years[i + 1]], [points[i], points[i + 1]], color=colormap(colors[i % len(colors)]),
             label='Red Bull Racing')

ax2.set_ylabel('Total Points')

# Set a common y-axis range for both subplots
min_y = min(min(max_points), min(points))
max_y = max(max(max_points), max(points)+150)
ax1.set_ylim(min_y, max_y)
ax2.set_ylim(min_y, max_y)

plt.tight_layout()
plt.show()
```
![0aa8f73e-ab22-42cc-acef-187ce5a2cc32](https://github.com/atnikola/f1-analysis-2023/assets/38530617/b4d6dad5-f3c2-4e90-994f-441d718b6dab)

Now let's take a look at the 'best' drivers for RedBull according to podium appearances:
```python
driverStandingDf = dataset['results_df'][(dataset['results_df']['constructorId'] == constructorId) &(dataset['results_df']['positionOrder'] <= 3)]
driverStandingDf.head(10)
```

```python
driverIdsAndNoOfPodiums = {}
def getDriverNameFromId(driverId):
    filtered_df = dataset['drivers_df'][(dataset['drivers_df']['driverId'] == driverId)][['forename','surname']]
    full_name = filtered_df['forename'] + " " + filtered_df['surname']
    return full_name.values[0]
for index, row in driverStandingDf.iterrows():
    try:
        driverIdsAndNoOfPodiums[row['driverId']] +=1
    except:
        driverIdsAndNoOfPodiums[row['driverId']] =1
driverIdsAndNoOfPodiums = dict(sorted(driverIdsAndNoOfPodiums.items(), key=lambda item: item[1], reverse=True))
names_ = [getDriverNameFromId(driverId) for driverId in driverIdsAndNoOfPodiums.keys()]
podiums_ = list(driverIdsAndNoOfPodiums.values())
print(driverIdsAndNoOfPodiums)
max_verstappen_driverId = 830
```

```python
# Create a horizontal bar chart
plt.barh(names_, podiums_)

# Add labels and title
plt.xlabel('Number of Podiums')
plt.ylabel('Driver Names')
plt.title('Number of Podiums by Driver')

# Show the plot
plt.show()

print(f"Top 3 Drivers of RedBull : ")
for i in range(3):
    print(f"{names_[i]} have {podiums_[i]} podiums")
```
![4c8712a5-c015-472d-8c4b-3094dd57b286](https://github.com/atnikola/f1-analysis-2023/assets/38530617/a941d4d9-9650-4697-b665-792c28cc54cc)

Top 3 Drivers of RedBull : 
Max Verstappen have 89 podiums
Sebastian Vettel have 65 podiums
Mark Webber have 41 podiums

Taking a closer look at Max's historical performance let's see his filtered df & look at his qualifying position vs his race finish position.
```python
quali_filtered_df = dataset["qualifying_df"][(dataset["qualifying_df"]['constructorId'] == constructorId) & (dataset["qualifying_df"]['driverId'] == max_verstappen_driverId)]
quali_filtered_df.head()
filtired_df = pd.merge(quali_filtered_df, dataset["results_df"], on= ['raceId','driverId','constructorId' ])
filtired_df.head(20)
```
<img width="1559" alt="2023-10-18 at 14 13 20" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/a23c383c-2d0f-4824-983c-a99f1a6abe8a">

```python
race_position = filtired_df['positionOrder'].to_list()
quali_position = filtired_df['position_x'].to_list()

# Create a hexagonal scatter plot
plt.hexbin(quali_position, race_position, gridsize=16, cmap='coolwarm', mincnt=1)

# Add color bar and labels
plt.colorbar(label='Count')
plt.xlabel('Quali. position')
plt.ylabel('Race Position')


# Show the plot
plt.show()
```
![97cbf54f-4c28-4671-a298-f711fe427da4](https://github.com/atnikola/f1-analysis-2023/assets/38530617/8804bf27-4f6b-4797-b409-7d9a58a4a941)

## [Classification](#Classification)

```python
# importing required libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
```
Combining dataframes:
```python
con1 = pd.merge(result_df, races_df, on ='raceId')
con2 = pd.merge(con1, drivers_df, on = 'driverId')
con3 = pd.merge(con2, driver_standings_df, on = 'driverId')
con4 = pd.merge(con3, constructor_df, on ='constructorId')
df = pd.merge(con4, stats_df, on ='statusId')
pd.get_option("display.max_columns",None)
df.head()
```
Cleaning
```python
df.count().isna()
```
<img width="246" alt="Screenshot 2023-10-18 at 15 10 47" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/8aa9a3d0-e8db-4ea9-9a62-621361f93627">

```python
df.describe()
```
<img width="1491" alt="Screenshot 2023-10-18 at 15 11 09" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/12169c87-71cf-4df1-9984-624cd8890b46">

```python
df = df.drop(['position_x'],1)

# changing of old column name to meaningful name 

col_name = {'number_x':'number','milliseconds':'timetaken_in_millisec','fastestLapSpeed':'max_speed',
 'name_x':'grand_prix','number_y':'driver_num','code':'driver_code','nationality_x':'nationality','name_y':'company',
 'raceId_x':'racerId','points_x':'points','position_y':'position'}

df.rename(columns=col_name,inplace=True)

df['driver_name'] = df['forename']+' '+df['surname']

df = df.drop(['forename','surname'],1)
```
```python
df['dob'] = pd.to_datetime(df['dob'])
from datetime import datetime
dates = datetime.today()-df['dob']
age = dates.dt.days/365

df['age'] = round(age)

pd.set_option('display.max_columns', None) # show all columns
df.head()
```
Changing dataype
```python
l = ['number','timetaken_in_millisec','fastestLap','rank','max_speed','driver_num']
for i in l:
    df[i] = pd.to_numeric(df[i],errors='coerce')
```

```python
#drop number
df.drop('driver_num',1,inplace=True)
```
```python
cat = []
num = []
for i in df.columns:
    if df[i].dtypes == 'O':
        cat.append(i)
    else:
        num.append(i)
```
Skewness:
Checking for skewness in the data, Skewness is used to check the normality of the data by ranging from -1 to 1.
-1 --> Left skewed
0 --> Normal distribution
1 --> Right skewed

```python
df.skew()
```
<img width="252" alt="Screenshot 2023-10-18 at 15 15 23" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/8c6cee4a-ef20-45c3-8cba-90e601c2c22a">

Outlier Treatment
```python
# outlier removal 

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).any(axis=1)]
df.head()
```
Heatmap
```python
plt.figure(figsize=(17,12))
sns.heatmap(df.corr(),annot=True)
plt.show()
```
![d24b5405-d48e-436e-be11-b03c1f3d29e2](https://github.com/atnikola/f1-analysis-2023/assets/38530617/88152f03-9b20-4f95-851e-09be5d009b9e)


```python
num.remove('dob')
num.remove('statusId')
```
KDE
```python
# kde plot for checking the normalization 

plt.figure(figsize=(15,50)) 
for i,j in zip(num,range(1,len(num)+1)):
    plt.subplot(11,2,j)
    sns.kdeplot(df[i],shade=True,color='lightblue')
    plt.grid(False)
plt.show()
```
![4128896a-439e-46e6-97f7-191604777425](https://github.com/atnikola/f1-analysis-2023/assets/38530617/8fc1e0bd-c058-4ac6-81b7-5d3637b3b625)

Encoding

I have choosen Label Encoding instead of OHE, because OHE will give you various new columns based on the unique value. 
* To represent a unicode string as a string of bytes is known as encoding.
* To convert a string of bytes to a unicode string is known as decoding.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in cat:
    df[i] = le.fit_transform(df[i])

x = df.drop('driver_name',1)
y = df.driver_name

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=101)
```
ML Algo

```python
# importing ML libraries 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
```
```python
clf = DecisionTreeClassifier(max_depth=5,random_state=1234)
clf.fit(xtrain, ytrain)
```
<img width="401" alt="Screenshot 2023-10-18 at 15 21 15" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/32c03640-3a8a-4b13-bfc1-cdabb2496b39">

```python
tree.export_text(clf)

fn = list(df.columns)
fn.remove('driver_name')
```

Modeling Data
```python
# classification ML algorithms 

lr = LogisticRegression(solver='sag')
dt = DecisionTreeClassifier()
rn = RandomForestClassifier()
knn = KNeighborsClassifier()
gb = GaussianNB()
sgd = SGDClassifier()
```
```python
li = [lr,sgd,knn,gb,rn,dt]
d = {}
for i in li:
    i.fit(xtrain,ytrain)
    ypred = i.predict(xtest)
    print(i,":",accuracy_score(ypred,ytest)*100)
    d.update({str(i):i.score(xtest,ytest)*100})
```
<img width="432" alt="Screenshot 2023-10-18 at 15 22 49" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/0e29ac21-295d-489e-a75a-91f24e5f58ef">

```python
plt.figure(figsize=(15, 7.5))
plt.title("Algorithm vs Accuracy", fontweight='bold')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(d.keys(),d.values(),marker='o',color='plum',linewidth=4,markersize=13,
         markerfacecolor='gold',markeredgecolor='slategray')
for x,y in zip(d.keys(),d.values()):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y),textcoords="offset points",xytext=(-3.75,5),ha='right')
plt.show()
```
![dd31b8c8-25ae-4f55-98e4-76b5f8f3aab9](https://github.com/atnikola/f1-analysis-2023/assets/38530617/2534585f-2886-43e2-b307-b63ffc73f11f)

As you can see, the accuracy of basic algorithm (logistic) and SGD is not good as expected. Since the data are skewed/denormalized, so it can be fixed by scaling the data.

MinMaxScaler
Each value in a feature, MinMaxScaler subtracts the minimum value in the feature and then divides by the range. The range is the difference between the original maximum and original minimum. MinMaxScaler preserves the shape of the original distribution.

```python
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(xtrain)
# transform training data
X_train_norm = norm.transform(xtrain)
# transform testing data
X_test_norm = norm.transform(xtest)
```
```python
li = [lr,sgd,rn,knn,gb,dt]
di = {}
for i in li:
    i.fit(X_train_norm,ytrain)
    ypred = i.predict(X_test_norm)
    print(i,":",accuracy_score(ypred,ytest)*100)
    di.update({str(i):i.score(X_test_norm,ytest)*100})
```
<img width="401" alt="Screenshot 2023-10-18 at 15 24 01" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/ab172d97-8ace-4c17-a62c-444b8d097928">

```python
plt.figure(figsize=(15, 7.5))
plt.title("Algorithm vs Accuracy", fontweight='bold')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(di.keys(),di.values(),marker='o',color='skyblue',linewidth=4,markersize=13,
         markerfacecolor='gold',markeredgecolor='black')
for x,y in zip(di.keys(),di.values()):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y),textcoords="offset points",xytext=(-3.75,5),ha='right')
plt.show()
```
![62394052-97b1-4953-918a-d9a67697f682](https://github.com/atnikola/f1-analysis-2023/assets/38530617/77779a46-ae39-4cf2-9503-737d78f02515)

As you can see the accuracy are getting high for Logistic Regression and SGDClassifier, both the algorithms are performing well from bottom low to 99% accuracy by scaling. Hence this prove the importance of normalizing the data.

StandardScaler
StandardScaler removes the mean and scales each feature/variable to unit variance. This operation is performed feature-wise in an independent way. StandardScaler can be influenced by outliers (if they exist in the dataset) since it involves the estimation of the empirical mean and standard deviation of each feature.

```python
from sklearn.preprocessing import StandardScaler
# fit scaler on training data
std = StandardScaler().fit(xtrain)
# transform train data
x_train_std = std.transform(xtrain)
# transform test data
x_test_std = std.transform(xtest)
```
```python
li = [lr,sgd,rn,knn,gb,dt]
dic = {}
for i in li:
    i.fit(x_train_std,ytrain)
    ypred = i.predict(x_test_std)
    print(i,":",accuracy_score(ypred,ytest)*100)
    dic.update({str(i):i.score(x_test_std,ytest)*100})
```
![37a269bc-70d2-4f7e-a935-89f711c29c3e](https://github.com/atnikola/f1-analysis-2023/assets/38530617/54ec3088-7c2e-4bfc-a3dd-bbc9ff268adb)

RobustScaler

Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile)

```python
from sklearn.preprocessing import RobustScaler
# fit scaler on train data
scaler = RobustScaler().fit(xtrain)
# transform train data
xtrain_scaled = scaler.transform(xtrain)
# transform test data
xtest_scaled = scaler.transform(xtest)
```

```python
li = [lr,sgd,rn,knn,gb,dt]
dics = {}
for i in li:
    i.fit(xtrain_scaled,ytrain)
    ypred = i.predict(xtest_scaled)
    print(i,":",accuracy_score(ypred,ytest)*100)
    dics.update({str(i):i.score(xtest_scaled,ytest)*100})
```
<img width="396" alt="Screenshot 2023-10-18 at 15 25 57" src="https://github.com/atnikola/f1-analysis-2023/assets/38530617/eeed137e-f0a5-4db9-93b1-f2ab2b68849f">

```python
plt.figure(figsize=(15, 7.5))
plt.title("Algorithm vs Accuracy", fontweight='bold')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(dics.keys(),dics.values(),marker='o',color='darkseagreen',linewidth=4,markersize=13,
         markerfacecolor='gold',markeredgecolor='black')
for x,y in zip(dics.keys(),dics.values()):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y),textcoords="offset points",xytext=(-3.75,5),ha='right')
plt.show()
```
![b467f8d0-5bce-47bc-87f1-e3147d7fdaf9](https://github.com/atnikola/f1-analysis-2023/assets/38530617/90b977e3-bf12-4517-82c9-aa823b6b35ac)















