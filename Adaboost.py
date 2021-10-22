from os import error
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import math
import sys

#decisiontree class
class decisiontreestump:
    def __init__(self):
        #classifier weight
        self.weight = None
        #best threshold value
        self.threshold = None
        #Prediction labels
        self.left_label = None
        self.right_label = None
    def return_labels(self,x_y):
        yield x_y[1]
    def calculate_entropy(self,collection):
        #find the unique labels in the collection
        label_count = Counter([next(self.return_labels(x_y)) for x_y in collection])
        total_instances = sum(label_count.values())
        entropy = 0
        for i in label_count:
            entropy += -(label_count[i]/total_instances)*math.log2(label_count[i]/total_instances)
        return entropy
    def threshold_gain_pair(self,thresholds,x,y):
        for v in thresholds:
            #values less than v
            left_collection = [(i[0],i[1]) for i in tuple(zip(self.x,self.y)) if i[0] <= v]
            right_collection = [(i[0],i[1]) for i in tuple(zip(self.x,self.y)) if i[0] > v]
            left_entropy = self.calculate_entropy(left_collection)
            right_entropy = self.calculate_entropy(right_collection)
            #calculate the weighted entropy for the split
            total_entropy = ((len(left_collection)/self.x.shape[0]) * left_entropy) + ((len(right_collection)/self.x.shape[0]) * right_entropy)
            gain = self.starting_entropy - total_entropy
            yield (v,gain)

    def find_threshold(self,x,y):
        self.x = x
        self.y = y
        #The entropy of the entire collection
        self.starting_entropy = self.calculate_entropy(list(zip(self.x,self.y)))
        #as it is bootstrapped data, there will be repeated tuples
        #it is a waste of computing resources to repeat entropy 
        # #calculations for each boundary, so we get the unique 
        #x values and sort it
        thresholds = sorted(np.unique(self.x),reverse = False)
        #split the data to include mid values where val%0.5 == 0
        thresholds = np.concatenate([np.arange(thresholds[0],thresholds[-1],step=0.5),[thresholds[-1]]])
        thresholds_gain_pair = self.threshold_gain_pair(thresholds,self.x,self.y)
        #find best treshold
        self.threshold = max(thresholds_gain_pair,key = lambda x:x[1])[0]
        #setting predicted outcome for each branch of the classifier
        self.left_label = Counter([i[1] for i in tuple(zip(self.x,self.y)) if i[0] <= self.threshold]).most_common(1)[0][0]
        self.right_label = Counter([i[1] for i in tuple(zip(self.x,self.y)) if i[0] > self.threshold]).most_common(1)[0][0]
    def make_prediction(self,x):
        if x <= self.threshold:
            return self.left_label
        else:
            return self.right_label

#This function does bootstrapping and adding the trained classifier to the 
#classifier dictionary
def bootstrap_train(classifiers_dict,i,data_df):
    bootstrap_sample = data_df.sample(data_df.shape[0],replace=True,weights=data_df["Weight"])
    #When the weight of an instance becomes very high compared to the other, the bootstrap_sample can have only one repeated 
    #instance. This causes the classifier incapable of making a split, as there is no feasible outcome for one branch
    #Hence when such a scenario occurs, we resample using the same weights, until the bootstrap_sample set at least one other instance
    while np.unique(bootstrap_sample["x"]).shape[0] == 1:
        bootstrap_sample = data_df.sample(data_df.shape[0],replace=True,weights=data_df["Weight"])
    #create decisiontree
    classifiers_dict[i] = decisiontreestump()
    classifiers_dict[i].find_threshold(bootstrap_sample["x"],bootstrap_sample["y"])
    prediction = list()
    for x in data_df["x"]:
        prediction.append(classifiers_dict[i].make_prediction(x))
    #calculating error rate
    return classifiers_dict,i,prediction,data_df

def buildAdaboostClassifier(classifiers,data_df,k):    
    for i in range(0,k):
        classifiers,i,prediction,data_df = bootstrap_train(classifiers,i,data_df)
        #calculate weight of classifier
        #If we have misclassified an instance with a higher weight, the contribution to error_rate by that instance increases
        #Hence your error rate increases
        error_rate = np.matmul(np.array([0 if i == j else 1 for i,j in zip(data_df["y"],prediction)]),data_df["Weight"])/(data_df.shape[0])
        #If the error_rate is greater than or equal to 0.5, we run the process again without changing weights of instances. 
        while error_rate >= 0.5:
            #As per Algorithm 5.7 in Introduction to Data Mining (TSK), if error_rate>0.5, the weights need to be reset to 1/N
            data_df["Weight"] = [1/data_df.shape[0]] * data_df.shape[0]
            classifiers,i,prediction,data_df = bootstrap_train(classifiers,i,data_df)
            error_rate = np.matmul(np.array([0 if i == j else 1 for i,j in zip(data_df["y"],prediction)]),data_df["Weight"])/(data_df.shape[0])
            

        data_df["Predicted"] = prediction
        if error_rate == 0:
            #In the case that error_rate is equal to zero we need to avoid division by zero
            #also such a classifier would be the ideal classifier, hence it's weight is set to 1. 
            alpha = 1
        else:
            alpha = 0.5 * np.log(((1- error_rate)/error_rate))
        #set classifier weight
        classifiers[i].weight = alpha
        #update weights of the tuples
        data_df["Weight"] = data_df["Weight"] * np.exp(-1 * alpha * data_df["y"] * data_df["Predicted"])
        #Normalizing the weights
        data_df["Weight"] = data_df["Weight"]/np.sum(data_df["Weight"])
    return classifiers,data_df

#Print weights of the classifiers
def printClassifierEquation(classifiers):
    #classifier equation
    equation = ""
    for i in classifiers:
        equation = equation + str(round(classifiers[i].weight,3)) + "*x +"
    print("Classifier Equation : sign[" + equation[:-2] + "]")

def printClassifierDetails(classifiers):
    print("Details of Each Classifier")
    details = ""
    for i in classifiers:
        threshold = classifiers[i].threshold
        details = details + "Classifier {} Details: ".format(i + 1) + "Classifier Weight:" + str(round(classifiers[i].weight,3)) +\
                            ", Classifier Threshold: " + str(threshold) +  "\nLeft Label <= {}".format(threshold) + " = {}".format(classifiers[i].left_label) +\
                                " and Right Label > {}".format(threshold) + " = {}".format(classifiers[i].right_label) + "\n"
    print(details)  


#predict the outcome for a given x value
def predict(classifiers,x):
    sign_value = 0
    for i in classifiers:
        sign_value += classifiers[i].weight*classifiers[i].make_prediction(x)
    if sign_value < 0:
        return -1
    else:
        return 1

#used to calculate accuracy
def calculate_accuracy(classifiers,x_values,label_list):
    if isinstance(label_list,list) and isinstance(x_values,list):
        predicted = list()
        for x in x_values:
            predicted.append(predict(classifiers,x))
        #calculating accuracy
        print("Classifier Accuracy : " + str(sum([1 if i==j else 0 for i,j in zip(predicted,label_list)])/len(label_list)))
        print("Predicted list: " + str(predicted))
    else:
        #only accepts list as values for x and y
        raise TypeError("Please enter x values and ground truth labels as a list")


def trainandpredictAdaboost(data_df,no_classifiers):
    classifiers = defaultdict(int)
    classifiers,data_df = buildAdaboostClassifier(classifiers,data_df,k=no_classifiers)
    #Prints the classifier equation
    printClassifierEquation(classifiers)
    printClassifierDetails(classifiers)
    #predict(classifiers,0)
    calculate_accuracy(classifiers,x_values = data_df["x"].tolist(),label_list = data_df["y"].tolist())


x = [0,1,2,3,4,5,6,7,8,9]
y = [1,1,-1,-1,-1,1,1,-1,-1,1]
data_dict = {}

data_dict["x"] = x
data_dict["y"] = y
data_dict["Weight"] = [1/len(data_dict["x"])]*len(data_dict["x"])
data_df = pd.DataFrame(data = data_dict)


trainandpredictAdaboost(data_df,no_classifiers=20)



