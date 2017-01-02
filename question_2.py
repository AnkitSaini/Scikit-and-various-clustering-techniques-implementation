import numpy as np
import math

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation

def final_labelling(clustering_labels):	
	#The labels 0 to 9 computed by the lustering methods are just the labels for
	#the cluster. They do not represent the digit that is represented by the cluster.
	#or is present in majority in the cluster. 
	#For example, the cluster with majority of images for number 5 can have label 0.
	#Hence this for loop finds the digit in majority in the cluster. 
	final_labels = [0]*len(clustering_labels)
	for i in range(0,10):
		#digit_counter is an array to keep count of the 
		#digit that has appeared the most number of times in
		#the target(or output) set of the input data
		digit_counter = [0]*10 
		#This loop counts the digits in cluster number 'i'
		for j in range(0,len(clustering_labels)):
			if clustering_labels[j] == i:
				d = target_values[j]
				digit_counter[d] += 1
		#The digit in majority in cluster i
		digit_maj = digit_counter.index(max(digit_counter))
		#This loop labels the cluster i with the digit that occurs in majority
		#in cluster 'i'
		for k in range(0,len(clustering_labels)):
			if clustering_labels[k] == i:
				final_labels[k] = digit_maj
			
	return final_labels

#Function to calculate the Fowlkes and Mallows index
def Fowlkes_Mallows_Index(cluster_tech_1, cluster_tech_2, target_values):
	index = 0 
	TP = 0 #True Positive = Points that are present in cluster_tech_1 and cluster_tech_2 and target_values
	FP = 0 #False Positive = Points present in target_values and cluster_tech_1 but not in cluster_tech_2
	FN = 0 #False Negative = Points present in target_values and cluster_tech_2 but not in cluster_tech_1
	TN = 0 #True Negative = Points present in target_values but not in cluster_tech_2 and cluster_tech_1
	for i in range(0,10):
		for j in range(0,len(target_values)):
			if target_values[j] == i and cluster_tech_1[j] == i and cluster_tech_2[j] == i:
				TP += 1 
			if target_values[j] == i and cluster_tech_1[j] == i and cluster_tech_2[j] != i:
				FP += 1 
			if target_values[j] == i and cluster_tech_2[j] == i and cluster_tech_1[j] != i:
				FN += 1 
			if target_values[j] == i and cluster_tech_1[j] != i and cluster_tech_2[j] != i:
				TN += 1 
	index = (TP*TP)/math.sqrt((TP + FP)*(TP + FN))
	print('True positive - ', + TP)
	print('False positive - ', + FP)
	print('False negative - ', + FN)
	print('True negative - ', + TN)
	print('Total = ', + (TP + FP + FN + TN))
	print('Index - ', + index)

def confusion_matrix(cluster_labels, target_values):
	conf_matrix = np.zeros((10,10))
	for i in range(0,10):
		for j in range(0, len(target_values)):
			if target_values[j] == i and cluster_labels[j] == i:
				conf_matrix[i][i] += 1 
			elif target_values[j] == i and cluster_labels[j] != i:
				digit = cluster_labels[j]
				conf_matrix[i][digit] += 1 
	print(conf_matrix)
		


np.random.seed(0)

#Importing dataset of handwritten digits
digits = datasets.load_digits()
#Extracting the input data and the target values 
data_input = digits.data
target_values = digits.target

n_sample, n_features = data_input.shape
n_digits = len(np.unique(digits.target))
print('='*50)
print("Number of features in the data set - ", + n_features)
print("Number of training examples in the data set - ", + n_sample)
print("Number of clusters - ", + n_digits)

#KMeans clustering using k-means++ initilization scheme so that the 
#centroids are initialized to be distant from each other
kmeans_cluster = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans_cluster.fit(data_input)
labels_kmeans = kmeans_cluster.labels_ 
#The labels 0-9 obtained from clustering are just labels for the cluster
#and do not represent the digit present in majority in the cluster hence 
#the function defined final_labelling() is used 
final_labels_kmeans = final_labelling(labels_kmeans)


#Agglomerative Clustering using the ward linkages. It gives the most regular
#cluster sizes. The other types of linkages available are average and complete linkage
agglomerative_cluster = AgglomerativeClustering(linkage='ward', n_clusters=10)
agglomerative_cluster.fit(data_input)
labels_agglomerative = agglomerative_cluster.labels_
final_labels_agglomerative = final_labelling(labels_agglomerative)


#Affinity Propagation 
affinity_cluster = AffinityPropagation(preference= -50000)
affinity_cluster.fit(data_input)
labels_affinity = affinity_cluster.labels_
final_labels_affinity = final_labelling(labels_affinity)

#Confusion Matrices
print('='*50)
print("Confusion Matrix for KMeans Clustering - ")
confusion_matrix(final_labels_kmeans, target_values)
print('='*50)
print("Confusion Matrix for Agglomerative Clustering - ")
confusion_matrix(final_labels_agglomerative, target_values)
print('='*50)
print("Confusion Matrix for Affinity Propagation - ")
confusion_matrix(final_labels_affinity, target_values)


#Fowlkes_Mallows_Index for Kmeans and agglomerative 
print('='*50)
print("Fowlkes and Mallows index for KMeans and Agglomerative Clustering - ")
Fowlkes_Mallows_Index(final_labels_kmeans, final_labels_agglomerative, target_values)
print('='*50)
print("Fowlkes and Mallows index for KMeans and Affinity Clustering - ")
Fowlkes_Mallows_Index(final_labels_kmeans, final_labels_affinity, target_values)
print('='*50)
print("Fowlkes and Mallows index for Agglomerative and Affinity Clustering - ")
Fowlkes_Mallows_Index(final_labels_affinity, final_labels_agglomerative, target_values)
print('='*50)
