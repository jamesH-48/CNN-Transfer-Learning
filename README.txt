CS 4372 Assignment 3
James Hooper ~ NETID: jah171230
Hritik Panchasara ~ NETID: hhp160130
------------------------------------------------------------------------------------------------------------------------------------
- For this assignment we utilized Google Colab
Numpy Version:  1.18.5
Python Version:  3.6.9
Tensorflow Version:  2.3.0
Keras Version:  2.4.0
- TO GET THE DATASET YOU MUST RUN THE !git COMMAND 
- IF YOURE NOT RUNNING IN COLAB YOU MUST RUN THIS SEPARATE TO GET THE DATASET IN THE LOCAL DIRECTORY
LINK TO DATASET: http://www.laurencemoroney.com/horses-or-humans-dataset/
https://www.tensorflow.org/datasets/catalog/horses_or_humans
held at: https://github.com/jamesH-48/horses_or_humans_test
------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS & PARAMETERS
if __name__ == '__main__':
~ Prints out the versions of the main packages used.
~ calls the preprocess function
~ calls the model function
def preprocess():
~ Downloads the data, Splits the data for training & testing & Augments the data
~ Also prints the images as needed
~ Augmentations: Rotate 90 degrees twice (upside down) and a color alteration (triples the training dataset)
def model(train_dataset, validation_dataset, test_dataset, labels_of_classes):
~ initializes the based (transfer learning) model and the new model created for the dataset specifically
~ Compiles and trains the model
~ Graphs and prints out the results (as well as test dataset results)
~ You can also print the initial model.evaluation if needed but it must be uncommented out (this was done to save time)
~ THE MAJOR PARAMTER CHANGES ARE HERE AND ARE SIMILAR TO THE LOG
