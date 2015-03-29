__author__ = 'hangma'

# Convert all png image files stored in ../data/nips_data/data_for_nips/dataset_i/rgb/ into:
# rgb.pkl.gz and grayscale.pkl.gz in ../data/

from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
import os


def convert(type):
    np.random.seed(1)

    Features = []
    Label = []
    for i in range(1 , 5):
        tempData = dir_to_dataset(os.path.join(os.path.split(__file__)[0], "../data/nips_data/data_for_nips/dataset_" + str(i) + "/" + "rgb" +"/*.png"), i, type)
        Features.extend(tempData[0])
        Label.extend(tempData[1])

    Data = Features, Label
    map(np.random.shuffle, Data)
    size_partition = int(Features.__len__()/5)

    # dataset divided into 3 parts 3:1:1.
    train_set = Data[0][ : 3*size_partition - 1], Data[1][ : 3*size_partition - 1]
    val_set = Data[0][3*size_partition : 4*size_partition -1], Data[1][3*size_partition : 4*size_partition -1]
    test_set = Data[0][4*size_partition : Features.__len__() - 1], Data[1][4*size_partition : Features.__len__() - 1]

    dataset = [train_set, val_set, test_set]

    f = gzip.open(os.path.join(os.path.split(__file__)[0], "../data/" + type + ".pkl.gz"),'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()


def dir_to_dataset(glob_files, label, type):
    print("Processing:\n\t %s"%glob_files)
    dataset = []
    labels = []
    print(glob(glob_files).__len__())
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        img = Image.open(file_name)
        if type == "rgb":
            pixels = list(img.getdata())
            pixels_flat = [(x/255.0) for sets in pixels for x in sets] # flatten RGBA values and normalize
            dataset.append(pixels_flat)
        if type == "grayscale":
            img.convert('LA') # tograyscale
            pixels = [x[0]/255.0 for x in list(img.getdata())] # only need the grayscale value and normalize
            dataset.append(pixels)
        labels.append(label)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    return np.array(dataset), np.array(labels)


def main():
    convert("rgb")
    convert("grayscale")

if __name__=='__main__':
	main()

