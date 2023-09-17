import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np



path = './merged'
all_files = glob.glob(os.path.join(path , "merged2019-*.csv"))

l = [ ]

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df.drop_duplicates(subset='Time', keep='first')
    EnergyProduction = df['Solar'].sum()+df['Wind'].sum()+df['Geothermal'].sum()+df['Biomass'].sum()+df['Biogas'].sum()+df['Small hydro'].sum()+df['Coal'].sum()+df['Nuclear'].sum()+df['Natural gas'].sum()+df['Large hydro'].sum()+df['Batteries'].sum()+df['Imports'].sum()+df['Other'].sum()
    EnergyDemand = df['Current demand'].sum()
    l.append([EnergyProduction, EnergyDemand])

ss = StandardScaler()
l = pd.DataFrame(ss.fit_transform(l), columns=['EnergyProduction', 'EnergyDemand'])

mms = MinMaxScaler()
mms.fit(l)
data_transformed = mms.transform(l)
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show(block=False)

km = KMeans(n_clusters=3)
model = km.fit(l)

colors=["red","blue","green","orange"]

def distance_from_center(income, age, label):
    '''
    Calculate the Euclidean distance between a data point and the center of its cluster.
    :param float income: the standardized income of the data point 
    :param float age: the standardized age of the data point 
    :param int label: the label of the cluster
    :rtype: float
    :return: The resulting Euclidean distance  
    '''
    center_income =  model.cluster_centers_[label,0]
    center_age =  model.cluster_centers_[label,1]
    distance = np.sqrt((income - center_income) ** 2 + (age - center_age) ** 2)
    return np.round(distance, 3)
l['label'] = model.labels_
l['distance'] = distance_from_center(l.EnergyProduction, l.EnergyDemand, l.label)

outliers_idx = list(l.sort_values('distance', ascending=False).head(10).index)
outliers = l[l.index.isin(outliers_idx)]
print(outliers)

# figure setting
plt.figure()
for i in range(np.max(model.labels_)+1):
    plt.scatter(l[model.labels_==i].EnergyProduction, l[model.labels_==i].EnergyDemand, label=i, c=colors[i], alpha=0.5, s=40)
plt.scatter(outliers.EnergyProduction, outliers.EnergyDemand, c='aqua', s=100)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], label='Centers', c="black", s=100)
plt.title("K-Means Clustering",size=20)
plt.xlabel("Energy Production")
plt.ylabel("Energy Demand")
plt.title('Energy Production vs Energy Demand')
plt.legend()
plt.show()