# f1-analysis-2023
**Andreas Nikolaidis** 

_October 2023_

- [Introduction](#introduction)
- [Redbull](#Redbull)

## [Introduction](#introduction)
In this notebook, we will look at the historical performance of drivers and constructors and compare the years.

## [Redbull](#Redbull)
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









