import os
import numpy as np
from PIL import Image
import random
import h5py
from sklearn.preprocessing import normalize

path = '/home/rakumar/DL_and_ML/train/'

img_folder_names = sorted(os.listdir(path))

def generate_dataset():
    image_file_names = []
    for name in img_folder_names:
        images = os.listdir(path+name)
        for img in images:
            image_file_names.append(path+name+'/'+img)

    allNpImg = []
    allLabels = []
    for imgfile in image_file_names:
        im = np.asarray(Image.open(imgfile))
        im = normalize(im, axis=1, norm='l1')
        im = im.reshape(28*28,)
        allNpImg.append(im)
        allLabels.append(int(imgfile.split('/')[-2]))

    res = list(zip(allNpImg, allLabels))
    random.shuffle(res)
    allNpImg, allLabels = zip(*res)

    hf = h5py.File('/home/rakumar/DL_and_ML/digitsDatasets.h5', 'w')
    hf.create_dataset('trainingset_image', data=allNpImg)
    hf.create_dataset('trainingset_label', data=allLabels)
    hf.close()

def load_dataset(path='/home/rakumar/DL_and_ML/digitsDatasets.h5'):
    train_dataset = h5py.File(path, 'r')
    train_set_x_orig = np.array(train_dataset['trainingset_image'][:])
    train_set_y_orig = np.array(train_dataset['trainingset_label'][:])
    return train_set_x_orig, train_set_y_orig

# generate_dataset()