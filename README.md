# KMN_on_Iris
#In this project , After downloading Iris Dataset From kaggle.com , for begining , the KMN method is applied on the dataset and then the efficiency of clustring method is calculated.
print(/n)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

		iris = load_iris()
		
		kmn = KMeans(n_clusters=3)
		kmn.fit(iris.data)
		labels = kmn.predict(iris.data)
		
		centroids = kmn.cluster_centers_
		plt.scatter(iris.data[:,0], iris.data[:,2], c=labels)
		plt.scatter(centroids[:,0],centroids[:,2],marker='x',s=150,alpha=0.5)
		plt.show()
		
	
		inertia_list = []
		for k in np.arange(1, 6):
			kmn = KMeans(n_clusters=k)
			kmn.fit(iris.data)
			inertia_list.append(kmn.inertia_)
		inertia_list
		
	plt.plot(np.arange(1,6),inertia_list,'ro-')
	plt.xlabel('number of clusters')
	plt.ylabel('Inertia')
	plt.show()
