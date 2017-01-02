import random
import csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

random.seed(0)

#Function to calculate f-measure
#The names of class_1 and class_2 are passed to the function as strings 
def f_measure(prediction_data, y_output_data, class_1, class_2):
	f_measure_index = 0
	TP = 0 #True Positive = class 1 member predicted as class 1
	TN = 0 #True Negative = class 2 member predicted as class 2 
	FP = 0 #False Positive = class 2 member predicted as class 1
	FN = 0 #False Negative = class 1 member predicted as class 2
	for i in range(0,len(prediction_data)):
		if y_output_data[i] == class_1 and prediction_data[i] == class_1:
			TP += 1
		if y_output_data[i] == class_2 and prediction_data[i] == class_2:
			TN += 1
		if y_output_data[i] == class_2 and prediction_data[i] == class_1:
			FP += 1
		if y_output_data[i] == class_1 and prediction_data[i] == class_2:
			FN += 1
	pr = TP/(TP + FP)
	re = TP/(TP + FN)
	f_measure_index = (2*pr*re)/(pr + re)
	print("True Positive - ", + TP)
	print("True Negative - ", + TN)
	print("False Positive - ", + FP)
	print("False Negative - ", + FN)
	print("Total - ", (TP+TN+FP+FN))
	print("Value of f-measure - ", + f_measure_index)
	

#reading the data from csv file
file = open('chronic_kidney_disease_full.csv')
data_read = csv.reader(file)
data = []
for row in data_read:
	data.append(row)

#Removing the first row of labels from the data 
data = data[1:]

#Shuffling the data in a random order because all the entries with ckd are grouped
#together and all the entries with notckd are grouped together 
random.shuffle(data)

#Separating the data into the Inputs and output for the classifier
x_input = []
y_output = []
for row in range(1,len(data)):
		x_input.append(data[row][0:24])#here 0:24 because it goes till 1 less that 24 
		y_output.append(data[row][24])

#Here we try to remove all the strings from input data, i.e. x_input
#This is done because the classifier was raising an error for all the strings
#present in the training data 
#normal, present, yes, good are replaced by 1 
#abnormal, notpresent, no, poor are replaced by 0
#'?' is replaced by 0
#and all the strings are converted to float 
for i in range(0,len(x_input)):
	for j in range(0,len(x_input[i])):
		if x_input[i][j] == '?':
			x_input[i][j] = 0
		elif x_input[i][j] == 'normal' or x_input[i][j] == 'present' or x_input[i][j] == 'yes' or x_input[i][j] == 'good' :
			x_input[i][j] = 1 
		elif x_input[i][j] == 'abnormal' or x_input[i][j] == 'notpresent' or x_input[i][j] == 'no' or x_input[i][j] == 'poor':
			x_input[i][j] = 0 
		else:
			x_input[i][j] = float(x_input[i][j])

#Splitting data into training and testing sets
data_split = int(0.8*len(x_input))
#Training set
x_input_train = x_input[:data_split]
y_output_train = y_output[:data_split]
#Testing set
x_input_test = x_input[data_split:]
y_output_test = y_output[data_split:]

print('='*50)
print("Length of input training data - ", + len(x_input_train))
#Support Vector Machine with linear kernel and default parameters
print('='*50)
print("Training data using SVC with linear kernel.....")
svc_linear_cls = svm.SVC(kernel='linear')
svc_linear_cls.fit(x_input_train, y_output_train)
svc_linear_prediction = svc_linear_cls.predict(x_input_train)
f_measure(svc_linear_prediction, y_output_train, 'ckd', 'notckd')
#print("Finished training.")

#Support Vector Machine with rbf kernel. It is the default kernel for 
#SVC hence we do not mention in parameters
print('='*50)
print("Training data using SVC with RBF kernel.....")
svc_rbf_cls = svm.SVC()
svc_rbf_cls.fit(x_input_train, y_output_train)
svc_rbf_prediction = svc_rbf_cls.predict(x_input_train)
f_measure(svc_rbf_prediction, y_output_train, 'ckd', 'notckd')
#print("Finished Training.")

#Random Forest classifier with default parameters
print('='*50)
print("Training data using Random Forest Classifier.....")
random_forest = RandomForestClassifier()
random_forest.fit(x_input_train, y_output_train)
random_forest_prediction = random_forest.predict(x_input_train)
f_measure(random_forest_prediction, y_output_train, 'ckd', 'notckd')
print('='*50)












