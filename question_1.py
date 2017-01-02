import csv
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1)

#Function to read data from csv file 
def read_data():
	file = open('realdata.csv')
	data_read = csv.reader(file)
	data = []
	for row in data_read:
		data.append(row)

	#removing the first column of row number from data
	for i in range(0,len(data)):
		data[i].pop(0)
	return data

#Function to initialize the centroids 	
def initialize_centroids(data, number_of_centroids):
	centroids = random.sample(data, number_of_centroids)
	return centroids

#Clustering labels the data points into separate clusters depending on their distance
#from the centroids 	
def clustering(data, centroids):
	distance = []
	cluster = []
	for i in range(0,len(data)):
		for j in range(0, len(centroids)):
			distance.append( ( (centroids[j][0] - data[i][0])**2 + (centroids[j][1] - data[i][1])**2 )**(1/2) )
		distance_min_centroid = distance.index(min(distance))
		cluster.append(distance_min_centroid)
		distance = []
	return cluster 

#Here we update the centroids by finding mean of the cluster 
def update_centroids(data, cluster, centroids):
	number_of_elements = 0
	sum_centroid_x = 0
	sum_centroid_y = 0
	new_centroids = list()
	for i in range(0,len(centroids)):
		new_centroids.append( list() )
	
	for i in range(0,len(new_centroids)):
		for j in range(0,len(data)):
			if cluster[j] == i:
				sum_centroid_x = sum_centroid_x + data[j][0]
				sum_centroid_y = sum_centroid_y + data[j][1]
				number_of_elements += 1 
		new_centroids[i].append(round(sum_centroid_x/number_of_elements,4))
		new_centroids[i].append(round(sum_centroid_y/number_of_elements,4))
		
		sum_centroid_x = 0
		sum_centroid_y = 0
		number_of_elements = 0

	return new_centroids	
	
def main(): 
	data = read_data()
	length = []
	width = [] 
	#Separating data into length and width
	for i in range(0,len(data)):
		length.append(float(data[i][0]))
		width.append(float(data[i][1]))
		data[i][0] = float(data[i][0])
		data[i][1] = float(data[i][1])
	centroids = initialize_centroids(data, number_of_centroids = 2)
	#initialized_centroids saves the co-ordinates of the centroids when they
	#were initialized
	initialized_centroids = list()
	for i in range(0,len(centroids)):
		initialized_centroids.append( list() )
	for i in range(0,len(centroids)):
		initialized_centroids[i].append(centroids[i][0])
		initialized_centroids[i].append(centroids[i][1])
	
	#Finding the cluster for each data point 
	cluster = clustering(data, centroids)
	
	#Threshold determines the limit to which we want to program to converge the 
	#centroids. We initialize threshold with a large number and then we terminate the 
	#loop when it get's below 
	threshold = 1000
	number_of_interation = 0 
	while(threshold > 0.002):
		
		new_centroids = update_centroids(data, cluster, centroids)
		
		print('='*50)
		number_of_interation += 1 
		print("Number of interation = ", + number_of_interation)
		print("Value of centroids - ")
		print(centroids)
		print("Value of updated centroids")
		print(new_centroids)
		print('='*50)
		
		cluster = clustering(data, centroids)
		
		threshold_distances = []
		for i in range(0,len(centroids)):
			threshold_distances.append( ( (centroids[i][0] - new_centroids[i][0])**2 + (centroids[i][1] - new_centroids[i][1])**2 )**(1/2) )
		
		threshold = max(threshold_distances)
		
		for i in range(0,len(centroids)):
			centroids[i][0] = new_centroids[i][0]
			centroids[i][1] = new_centroids[i][1]
	
	cluster_1_x = []
	cluster_1_y = []
	cluster_2_x = []
	cluster_2_y = []
	for i in range(0,len(data)):
		if cluster[i] == 0:
			cluster_1_x.append(data[i][0])
			cluster_1_y.append(data[i][1])
		if cluster[i] == 1:
			cluster_2_x.append(data[i][0])
			cluster_2_y.append(data[i][1])
	plt.figure(1)
	#Plotting the entire data
	plt.plot(length,width,'bo')
	plt.xlabel("Length")
	plt.ylabel("Width")
	plt.title("KMeans clustering")
	
	#Plotting cluster 1 and 2
	plt.plot(cluster_1_x,cluster_1_y,'yo')
	plt.plot(cluster_2_x,cluster_2_y,'ko')
	
	#Plotting the centroid with which we started the kmeans
	plt.plot(initialized_centroids[1][0],initialized_centroids[1][1],'r^', ms = 10)
	plt.plot(initialized_centroids[0][0],initialized_centroids[0][1],'r^', ms = 10)
	
	#plotting final centroid 
	plt.plot(new_centroids[0][0],new_centroids[0][1],'go', ms = 10)
	plt.plot(new_centroids[1][0],new_centroids[1][1],'go', ms = 10)
	
	#plt.text(new_centroids[0][0],new_centroids[0][1], 'Cluster 1')
	plt.annotate('Cluster 1', xy=(1.2, 0.7), xytext=(1.6, 0.8),arrowprops=dict(facecolor='black', shrink=0.05))
	plt.annotate('Cluster 2', xy=(2.6, 0.5), xytext=(1.5, 0.6),arrowprops=dict(facecolor='black', shrink=0.05))
	
	plt.show()
	

if __name__ == "__main__":
	main()
	
	
