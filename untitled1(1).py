import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import platform

def preprocess():
    (dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
        name='rock_paper_scissors',
        data_dir='tmp',
        with_info=True,
        as_supervised=True,
        split=['train', 'test'],
    )

    for image, label in dataset_train_raw.take(1):
        print(image.shape, label)

    # Print out info of the dataset
    print(dataset_info)

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
    print('Labels of Classes: ', labels_of_classes(0), labels_of_classes(1), labels_of_classes(2))


    # Plot Some Images of the Dataset for viewing
    plt.figure(figsize=(8, 8))
    plt_indx = 0
    for features in dataset_train_raw.take(8):
        (image, label) = features
        plt_indx += 1
        plt.subplot(3, 4, plt_indx)
        label = labels_of_classes(label.numpy())
        plt.title('Label: %s' % label)
        plt.imshow(image.numpy())


if __name__ == '__main__':
    # Print out main versions of packages used
    print("Numpy Version: ", np.version.version)
    print("Python Version: ", platform.python_version())
    print("Tensorflow Version: ", tf.__version__)
    print("Keras Version: ", tf.keras.__version__)

    '''
    Numpy Version:  1.18.5
    Python Version:  3.6.9
    Tensorflow Version:  2.3.0
    Keras Version:  2.4.0
    '''

    # Pre-process data
    preprocess()
