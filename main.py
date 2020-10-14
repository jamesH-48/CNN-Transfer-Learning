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
    #print_data(raw_train_data, labels_of_classes, 'cyan', images_only=False)

    # Spacing
    print()

    # Scale data to be between 0-1
    scaled_train_data = raw_train_data.map(format_image)
    scaled_test_data = raw_test_data.map(format_image)

    # Plot Some Images of the Dataset for viewing
    # Print Data of First Image
    # Scaled 
    #print_data(scaled_train_data, labels_of_classes, 'violet', images_only=False)

    '''
      Augment Data
    '''
    train_data = augmentation(scaled_train_data)
    
    # Shuffle Data for print
    train_data_full = train_data.shuffle(5135)
    print("Length of Data", len(list(train_data)))
    #print_data(train_data, labels_of_classes, 'white', images_only=True)

    # Split Train Dataset into Train & Validation 
    split_index = int(5135*.3)
    valid_data = train_data_full.take(split_index)
    train_data = train_data_full.skip(split_index)

    print('train dataset size:', len(list(train_data)), '\n')
    print('valid dataset size:', len(list(valid_data)), '\n')

    return train_data, valid_data, scaled_test_data, dataset_info

# Image Augmentation to add more images to dataset
def augmentation(data):
    new_data = data
    temp_data = data

    # Print length of data aka number of examples
    # Should be 1027
    # print("Length of Data", len(list(new_data)))

    '''
      Rotate Data
        - This should quadruple our data size by adding more images to the set
    '''
    for i in range(3):   
      temp_data = temp_data.map(aug_rot90)
      new_data = new_data.concatenate(temp_data)

    # Print length of data aka number of examples
    # Should be 1027*4 = 4108
    # print("Length of Data", len(list(new_data)))

    '''
      Randomly Change the Hue of the Data
        - This should add 1027 more images to the dataset
    '''
    temp_data = data.map(aug_rand_hue)
    new_data = new_data.concatenate(temp_data)

    # Print length of data aka number of examples
    # Should be 4108 + 1027 = 5135
    # print("Length of Data", len(list(new_data)))

    return new_data

def aug_rot90(image, label):
    image = tf.image.rot90(image, 1)
    return image, label

def aug_rand_hue(image, label):
    image = tf.image.random_hue(image, max_delta=.1)
    return image, label

# print out images and data for comparisson 
def print_data(data, labels_of_classes, color_choice, images_only):
    # Plot Some Images of the Dataset for viewing
    # Original Scale
    plt.figure(figsize=(16, 16))
    plt_indx = 0
    font = {
        'color':  color_choice,
        'weight': 'normal',
        'size': 16,
        }

    for features in data.take(16):
        (image, label) = features
        plt_indx += 1
        plt.subplot(4, 4, plt_indx)
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

def model(train_data, valid_data, scaled_test_data, dataset_info):
    # BATCHES
    BSIZE = 32
    train_data = train_data.batch(batch_size = BSIZE)
    # for better performance
    train_data = train_data.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    valid_data = valid_data.batch(batch_size = BSIZE)

    print(train_data)
    print(valid_data)

    # New Input
    input_new = Input(shape=(300,300,3), name = 'Image_Input')
    
    # VGG16 Base Model
    base_model_VGG16 = tf.keras.applications.VGG16(
        include_top = False,
        weights = 'imagenet',
        input_tensor = input_new,
        pooling = 'avg'
    )

    # Freeze VGG16 Base Model
    base_model_VGG16.trainable = False
    
    # Grab the final layer to connect more layers after
    final_layer = base_model_VGG16.layers[-1]
    

    # Connect Flatten, Dense, and Softmax Output Layers
    x = Flatten(name='flatten')(final_layer.output)
    x = Dense(4096, activation ='relu', name = 'FC1')(x)
    x = Dense(4096, activation ='relu', name = 'FC2')(x)
    x = Dense(2, activation = 'softmax', name = 'Predictions')(x)

    # Create Model
    model = Model(inputs=input_new, outputs=x, name="NewModel")

    # Print Summary of Model
    print(model.summary())

    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001), 
                  loss = tf.keras.losses.sparse_categorical_crossentropy, 
                  metrics = ['accuracy']
    )

    print("len: ", len(model.trainable_variables))
   
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc',
                                 verbose=1, save_best_only=True, 
                                 save_weights_only=False,
                                 mode='auto', period=1)
    
    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=20, verbose=1, mode='auto')

    history = model.fit(
        x = train_data,
        validation_data = valid_data,
        epochs = 10,
        steps_per_epoch = len(list(train_data))//BSIZE,
        validation_steps = len(list(valid_data))//BSIZE,
        verbose = 1,
        callbacks=[checkpoint,early]
    )

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
    train_data, valid_data, scaled_test_data, dataset_info = preprocess()

    # Model and Results
    # This will include batch manipulation as well
    model(train_data, valid_data, scaled_test_data, dataset_info)
