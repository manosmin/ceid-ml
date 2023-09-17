import pandas as pd
import matplotlib.pyplot as plt

path = 'merged2021-10-30.csv'
# Read merged csv
df = pd.read_csv('merged/'+path)

# Display top 5 rows
print(df.head())

# Display all columns and their data types
print(df.info())

# This function shows you some basic descriptive statistics for all numeric columns:
print(df.describe())

def f(x):
    a = x['Current demand'].sum()
    b = x['Time'].unique()
    return pd.Series([a, b], index=['Current demand', 'Time'])


df1 = df.groupby(['Time']).apply(f)

t1 = []
t2 = []

for i in range(0, 288):
    t2.append(df1.iloc[i][1])
    t1.append(df1.iloc[i][0])

data = {'Current demand': t1, 'Time': t2}

# Getting sum of each column
df.loc['sum'] = df.sum(numeric_only=True, axis=0)

# Storing column names into an array
data3 = [
        'Solar', 
        'Wind', 
        'Geothermal', 
        'Biomass', 
        'Biogas', 
        'Small hydro', 
        'Coal',
        'Nuclear', 
        'Natural gas', 
        'Large hydro', 
        'Batteries', 
        'Imports', 
        'Other'
        ]

# Storing sum column values into an array
index = [
        df.loc['sum'][1], 
        df.loc['sum'][2], 
        df.loc['sum'][3], 
        df.loc['sum'][4], 
        df.loc['sum'][5], 
        df.loc['sum'][6], 
        df.loc['sum'][7], 
        df.loc['sum'][8], 
        df.loc['sum'][9], 
        df.loc['sum'][10], 
        df.loc['sum'][11], 
        df.loc['sum'][12],
        df.loc['sum'][13]
        ]

# Plotting dataframe
df = pd.DataFrame({'Sources': data3, 'Amount': index})
df.groupby(['Sources']).sum().plot(kind='pie', y='Amount', autopct='%.2f', figsize=(8, 8), title=path)
df2 = pd.DataFrame(data, columns=['Current demand', 'Time'])
df2.plot(x='Time', y='Current demand', kind='line', title=path)
plt.show()