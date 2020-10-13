# Run on Google Colab
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import platform

def preprocess():
    (raw_train_data, raw_test_data), dataset_info = tfds.load(
        name='horses_or_humans',
        with_info=True,
        as_supervised=True,
        split=['train', 'test'],
    )

    # Print out info of the dataset
    # print(dataset_info)

    # Get Number of Train and Test examples & Amount of Classes
    train_examples_num = dataset_info.splits['train'].num_examples
    test_examples_num = dataset_info.splits['test'].num_examples
    classes_amount = dataset_info.features['label'].num_classes

    print('Number of Train Examples:', train_examples_num)
    print('Number of Test Examples:', test_examples_num)
    print('Amount of Classes:', classes_amount)

    # Get image shape
    image_shape = dataset_info.features['image'].shape
    print('Image Shape: ', image_shape)

    # Print Labels (Classes)
    labels_of_classes = dataset_info.features['label'].int2str
    print('Labels of Classes: ', labels_of_classes(0), labels_of_classes(1))

    # Spacing
    print()

    # Plot Some Images of the Dataset for viewing
    # Print Data of First Image
    # Original Scale
    print_data(raw_train_data, labels_of_classes, 'cyan', images_only=False)

    # Spacing
    print()

    # Scale data to be between 0-1
    train_data = raw_train_data.map(format_image)
    test_data = raw_test_data.map(format_image)

    # Plot Some Images of the Dataset for viewing
    # Print Data of First Image
    # Scaled 
    print_data(train_data, labels_of_classes, 'violet', images_only=False)

# print out images and data for comparisson 
def print_data(data, labels_of_classes, color_choice, images_only):
    # Plot Some Images of the Dataset for viewing
    # Original Scale
    plt.figure(figsize=(10, 10))
    plt_indx = 0
    font = {
        'color':  color_choice,
        'weight': 'normal',
        'size': 16,
        }

    for features in data.take(9):
        (image, label) = features
        plt_indx += 1
        plt.subplot(3, 3, plt_indx)
        label = labels_of_classes(label.numpy())
        plt.title(label, fontdict=font)
        plt.imshow(image.numpy())

    if not images_only:
      # Print out original scale of data
      for features in data.take(1):
        (image, label) = features
        print("Data Values for first image")
        print("Shape: ", image.numpy().shape)
        print("Label: ", labels_of_classes(label.numpy()))
        print("Original Scale: ", image.numpy())


# We need to scale the data to allow for use in the model
def format_image(image, label):
  # cast as float
  image = tf.cast(image, tf.float32)
  # scale values
  image = image*1/255.0
  # resize to correct shape
  image = tf.image.resize(image, (300,300))
  return image, label

if __name__ == '__main__':
    # Print out main versions of packages used
    print("Numpy Version: ", np.version.version)
    print("Python Version: ", platform.python_version())
    print("Tensorflow Version: ", tf.__version__)
    print("Keras Version: ", tf.keras.__version__)
    print()

    '''
    Numpy Version:  1.18.5
    Python Version:  3.6.9
    Tensorflow Version:  2.3.0
    Keras Version:  2.4.0
    '''

    # Pre-process data
    preprocess()
