import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import platform
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def preprocess():
    train_dir = 'C://Users/Atlas/PycharmProjects/A3_CS4372/thisone/horse-or-human'
    validation_dir = 'C://Users/Atlas/PycharmProjects/A3_CS4372/thisone/validation-horse-or-human'

    BATCH_SIZE = 32
    IMG_SIZE = (300, 300)

    # Get the train & validation datasets
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    print('train dataset size:', len(list(train_dataset)), '\n')
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                                    validation_dir,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE)
    # Get class labels
    labels_of_classes = train_dataset.class_names
    print("Labels of Classes: ", labels_of_classes)
    print_images(train_dataset, labels_of_classes, 'cyan')
    print("Original Image Data: ")
    print_image_data(train_dataset, labels_of_classes)

    # Spacing
    print()

    # Make Test Data Set
    # This should leave 1 batch for testing (32 Images)
    # For this project we only need 25 images
    VAL_BATCHES = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(VAL_BATCHES // 5)
    validation_dataset = validation_dataset.skip(VAL_BATCHES // 5)
    print("Number of validation batches: ", tf.data.experimental.cardinality(validation_dataset))
    print("Number of test batches: ", tf.data.experimental.cardinality(test_dataset))

    # We need to configure the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Augment Data
    train_dataset = augmentation(train_dataset, labels_of_classes)
    # print('train dataset size:', len(list(train_dataset)), '\n')

    plt.show()
    return train_dataset, validation_dataset, test_dataset

# Image Augmentation to add more images to dataset
def augmentation(data, labels_of_classes):
    new_data = data
    temp_data = data

    # Print length of data aka number of examples
    # Should be 1027
    # print("Length of Data", len(list(new_data)))

    '''
      Rotate Data
        - This should double our data size by adding more images to the set
    '''
    temp_data = temp_data.map(aug_rot90)
    new_data = new_data.concatenate(temp_data)
    print_images(temp_data, labels_of_classes, 'violet')
    # Print length of data aka number of examples
    # Should be 1027*4 = 4108
    # print("Length of Data", len(list(new_data)))

    '''
      Randomly Change the Hue of the Data
        - This should add 1027 more images to the dataset
    '''
    temp_data = data.map(aug_rand_hue)
    print_images(temp_data, labels_of_classes, 'black')
    new_data = new_data.concatenate(temp_data)

    # Print length of data aka number of examples
    # Should be 4108 + 1027 = 5135
    # print("Length of Data", len(list(new_data)))

    return new_data

def aug_rot90(image, label):
    # upside down
    image = tf.image.rot90(image, 2)
    return image, label

def aug_rand_hue(image, label):
    image = tf.image.random_hue(image, max_delta=.1)
    return image, label

# print out images for comparisson
def print_images(data, labels_of_classes, color_choice):
    # Plot Some Images of the Dataset for viewing
    # Original Scale
    plt.figure(figsize=(16, 16))
    font = {
        'color':  color_choice,
        'weight': 'normal',
        'size': 16,
        }

    for images, labels in data.take(1):
        for i in range(16):
            plt.subplot(4, 4, i+1)
            label = labels_of_classes[labels[i]]
            plt.title(label, fontdict=font)
            plt.imshow(images[i].numpy().astype("uint8"))

# print out data for a single image
def print_image_data(data, labels_of_classes):
    for image, label in data.take(1):
        print("Shape of Image: ", image[0].numpy().shape)
        print("Label of Image: ", labels_of_classes[label[0]])
        print("Original Scale: ", image[0].numpy())

def model(train_dataset, validation_dataset, test_dataset):
    # Scale values for Base Model
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    # VGG16 Base Model
    base_model_VGG16 = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    print(base_model_VGG16.summary())
    base_model_VGG16.trainable = False

    inputs = tf.keras.Input(shape=(300,300,3))
    x = preprocess_input(inputs)
    x = base_model_VGG16(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2, activation='softmax', name='Predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    print(model.summary())

    initial_epochs = 4
    #loss0, accuracy0 = model.evaluate(validation_dataset)
    #print("initial loss: {:.2f}".format(loss0))
    #print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    # data results for plotting
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Training & Validation Accuracy
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label='Training Accuracy', color='green')
    plt.plot(val_accuracy, label='Validation Accuracy', color='cyan')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    # Training & Validation Loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss', color='green')
    plt.plot(val_loss, label='Validation Loss', color='cyan')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.show()

    # Use Test Dataset for Final Predictions
    test_image_batch, test_label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(test_image_batch).flatten()
    labels_of_classes = train_dataset.class_names

    print("Predictions for Test Dataset:\n", predictions.numpy())
    print("Labels:\n", test_label_batch)

    plt.figure(figsize=(40,40))
    for i in range(32):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(test_image_batch[i].astype("uint8"))
        plt.title(labels_of_classes[predictions[i]])
        plt.axis("off")
    plt.show()

if __name__ == '__main__':
    # Print out main versions of packages used
    print("Numpy Version: ", np.version.version)
    print("Python Version: ", platform.python_version())
    print("Tensorflow Version: ", tf.__version__)
    print("Keras Version: ", tf.keras.__version__)
    print()

    # Call Pre-Process Function
    train_dataset, validation_dataset, test_dataset = preprocess()

    # Call Model Function
    model(train_dataset, validation_dataset, test_dataset)
