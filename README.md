# Leveraging Mutants for Automatic Prediction of MetamorphicRelations using Machine Learning 

A python implementation for the paper "Leveraging Mutants for Automatic Prediction of MetamorphicRelations using Machine Learning" (Maltesque 2019)

createPickle.py: Takes the Dot files and their corresponding class labels of a corresponding MR as input and generates a graph pickle object out of it. This graph pickle could be loaded by other programs for applying graph algorithms on it. 

get_ROC-py: Takes the graph pickle as input and perform graph ML algorihtms on it to classiyfing it to its MR class. Later it provides a ROC metric containing the classifier accuracy details. 

my_functions.py: Used to calculate the Random Walk Kernel (RWK) between two graphs. 



