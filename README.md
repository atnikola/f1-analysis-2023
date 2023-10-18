# f1-analysis-2023
**Andreas Nikolaidis** 

_October 2023_

- [Introduction](#introduction)
- [Exploratory Analysis](#exploratory_analysis)
- [Correlations & Descriptive Statistics](#descriptive)
- [Principal Component Analysis (PCA)](#pca)
- [Cross Validation & Regression Analysis](#cv-ra)
- [Conclusion](#conclusion)

## [Introduction](#introduction)
In this notebook, we will look at the historical performance of drivers and constructors and compare the years.

## [Exploratory Analysis](#exploratory_analysis)
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











