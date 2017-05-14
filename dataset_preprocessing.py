import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
from sklearn.utils import shuffle

out_img_dir = './output_images'

def load_images():
    """
    Loads all images

    :return: (cars) The cars data, (noncars) The non-cars data
    """
    cars = []
    noncars = []

    cars.extend(glob.glob('./vehicles/GTI_Far/*.png'))
    cars.extend(glob.glob('./vehicles/GTI_MiddleClose/*.png'))
    cars.extend(glob.glob('./vehicles/GTI_Left/*.png'))
    cars.extend(glob.glob('./vehicles/GTI_Right/*.png'))
    cars.extend(glob.glob('./vehicles/KITTI_extracted/*.png'))
    noncars.extend(glob.glob('./non-vehicles/GTI/*.png'))
    noncars.extend(glob.glob('./non-vehicles/Extras/*.png'))

    return cars, noncars


def train_valid_test_split(data, train_pct=0.7, test_pct=0.1):
    """
    Splits data into training, validation, and test sets

    :param data: The dataset to split
    :param train_pct: The percentage to allocate for training
    :param test_pct: The percentage to allocate for testing
    :return: (train) The training data, (test) The testing data,
             (valid) The validation data
    """
    # Shuffle the data
    shuffled = shuffle(data)

    # Get the length of the dataset
    length = len(data)

    # Split the data into train, valid, test and return
    train_end = int(train_pct * length)
    test_end = train_end + int(test_pct * length)
    train = data[0:train_end]
    test = data[train_end:test_end]
    valid = data[test_end:-1]
    return train, test, valid


def save_to_pickle(cars_train, cars_test, cars_valid, noncars_train,
                   noncars_test, noncars_valid, filename='data.p'):
    """
    Saves the data to a pickle file

    :param filename: The name of the pickle file
    :param cars_train: The cars training set
    :param cars_test: The cars test set
    :param cars_valid: The cars validation set
    :param noncars_train: The noncars training set
    :param noncars_test: The noncars test set
    :param noncars_valid: The noncars validation set
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(
                {
                    'cars_train': cars_train,
                    'cars_test': cars_test,
                    'cars_valid': cars_valid,
                    'noncars_train': noncars_train,
                    'noncars_test': noncars_test,
                    'noncars_valid': noncars_valid
                },
                file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Pickle save failed:', e)
        raise

    print('Pickle file saved')


def preprocess():
    """
    The dataset preprocessing pipeline
    """
    # Load the images
    cars, noncars = load_images()

    # Split the datasets into training, validation, and test
    cars_train, cars_test, cars_valid = train_valid_test_split(cars)
    noncars_train, noncars_test, noncars_valid = train_valid_test_split(noncars)

    # Print the length of the datasets
    print('Cars: Total: %s, Train: %s, Test: %s, Valid: %s' % (len(cars), len(cars_train),
                                                               len(cars_test), len(cars_valid)))
    print('Noncars: Total: %s, Train: %s, Test: %s, Valid: %s' % (len(noncars), len(noncars_train),
                                                                  len(noncars_test), len(noncars_valid)))

    # Generate histograms of dataset sizes
    xlabels = ['Train', 'Valid', 'Test']
    pos = np.arange(len(xlabels))
    data_cars = [len(cars_train), len(cars_valid), len(cars_test)]
    data_noncars = [len(noncars_train), len(noncars_valid), len(noncars_test)]
    plt.figure()
    plt.bar(pos, data_cars, align='center')
    plt.xticks(pos, xlabels)
    plt.ylabel('Data Points')
    plt.title('Cars')
    plt.savefig(out_img_dir + '/preprocess_hist_cars')
    plt.figure()
    plt.bar(pos, data_noncars, align='center')
    plt.xticks(pos, xlabels)
    plt.ylabel('Data Points')
    plt.title('Non-Cars')
    plt.savefig(out_img_dir + '/preprocess_hist_noncars')

    # Plot an example of a car vs. a noncar
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(plt.imread(cars_train[0]))
    ax2.imshow(plt.imread(noncars_train[0]))
    ax1.set_title('Car example')
    ax2.set_title('Non-car example')
    plt.savefig(out_img_dir + '/preprocess_car_vs_noncar')

    # Save data to pickle
    save_to_pickle(cars_train, cars_test, cars_valid, noncars_train, noncars_test, noncars_valid)


preprocess()