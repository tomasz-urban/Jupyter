# Clustering

`a) importing libraries`
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()

`b) categorical variables`
df_map = data.copy()
df_map['categorical_variable'] = df_map['categorical_variable'].map({'val_1': 0, 'val_2': 1, 'val_3': 2})
df_map

`c) selecting features`
x = data.copy()

#or if we have to choose some columns from the dataset we can do this like this:
x = data.iloc[:,1:3] #all the rows, column with indexes 1 and 2
x

`d) elbow method`
wcss = [] #within clusters sum of squares
n = 10 # how many clusters we want to check
for i in range(1,n):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

#Ploting the elbow method results
clusters_number = range(1,n) 
plt.plot(clusters_number, wcss);

`e) clustering`
kmeans = KMeans(2) # 2 clusters
kmeans.fit(x)
#kmeans.get_params() # if we can't see the parameters we can check them like that

`f) clustering results`
clusters = kmeans.fit_predict(x)

clusters_df = df_map.copy()
clusters_df['Clusters'] = clusters
clusters_df

`g) ploting results`
plt.scatter(clusters_df['var_1'], clusters_df['vr_2'],c=clusters_df['Clusters'], cmap = 'rainbow')
plt.show() 

`h) standardization`
#If one variable has much bigger values we might have to use standardization. In this case KMeans is not considering variable with little values and divides only using #the one with bigger values. To change this we are using standardization for KMeans to treat both variables equally.

from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled

#After that we get back to `d)elbow method` and use x_scaled instead of x

`i) heatmap`
#If we want to check some hierarchy in the data (for example countries in the continents clusters) we can use heatmap
#x - we use variables we want to check (compare) - like with the countries Longitude and Latitude
sns.clustermap(x, cmap='mako');

