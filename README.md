# Adaboost-Classifier-From-Scratch

Using multiple weak classifiers to create a strong classifier has proven to show better results in classification tasks. 
Approaches to this include:
- Bagging
- Boosting
- Ensembles such as Random forests

In this repository, I present an Adaboost classifier from scratch, using a decision tree stump as the classifier
Currently it is made to work on one attribute. Later improvements will be made to make it scalable for multiple attributes. 

The final prediction uses the following equation: 

sign[α1C1(x)+ α2C2(x) + α3C3(x) + . . .], where Ci(x) is the base classifier and αi
is the weight of base classifier

The script prints out the following in your console:
- Strong Classifier Equation
- Details of each weak classifier
- Accuracy of prediction
- Predicted result

