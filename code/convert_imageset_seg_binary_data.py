__author__ = 'hangma'

# Convert all png image files stored in ../data/nips_data/data_for_nips/dataset_i/rgb/ into:
# rgb.pkl.gz and grayscale.pkl.gz in ../data/

from PIL import Image
from StringIO import StringIO
from urllib import urlopen
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

    tempData = dir_to_dataset(0, type, "wo_object")
    Features.extend(tempData[0])
    Label.extend(tempData[1])

    tempData = dir_to_dataset(1, type, "occluded")
    Features.extend(tempData[0])
    Label.extend(tempData[1])

    tempData = dir_to_dataset(2, type, "with_object")
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

    f = gzip.open(os.path.join(os.path.split(__file__)[0], "../data/" + type + "_seg_binary_data.pkl.gz"),'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()


def dir_to_dataset(label, type, folder_name):
    dataset = []
    labels = []
    for num_seg in range (0, 6):
        for i in range(0, 205):
            url_rgb = urlopen("http://www-clmc.usc.edu/~bharath/nips_data/binary_data/" + folder_name + "/seg_" + str(num_seg) + "/rgb_image_" + str(i) + ".png")
            if url_rgb.headers.maintype == 'image':
                img_rgb = Image.open(StringIO(url_rgb.read()))
                img_rgb = img_rgb.resize((64, 48), Image.ANTIALIAS)
                url_depth = urlopen("http://www-clmc.usc.edu/~bharath/nips_data/binary_data/" + folder_name + "/seg_" + str(num_seg) + "/depth_image_" + str(i) + ".png")
                if url_depth.headers.maintype != 'image':
                    print(folder_name + " seg " + str(num_seg) + " depth image " + str(i) + " is not presented while rgb image is presented")
                    continue
                img_depth = Image.open(StringIO(url_depth.read()))
                img_depth = img_depth.resize((64, 48), Image.ANTIALIAS)
                pixels_depth = list(img_depth.getdata())
                if type == "rgb":
                    pixels = list(img_rgb.getdata())
                    pixels_flat = [(x/255.0) for sets in pixels for x in sets] + pixels_depth # flatten RGBA values and normalize
                    dataset.append(pixels_flat)
                if type == "grayscale":
                    img_rgb.convert('LA') # tograyscale
                    pixels = [x[0]/255.0 for x in list(img_rgb.getdata())] + pixels_depth # only need the grayscale value and normalize
                    dataset.append(pixels)
                labels.append(label)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    return np.array(dataset), np.array(labels)


def main():
    convert("rgb")
    #convert("grayscale")

if __name__=='__main__':
	main()

