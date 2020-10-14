import struct as st
import numpy as np
import os

def unpack(image_file, label_file):
    '''
        Code taken from https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
    '''

    # Open the IDX file in readable binary mode
    filename = {'images' : image_file ,'labels' : label_file}
    train_imagesfile = open(filename['images'],'rb')

    # Read the magic number
    train_imagesfile.seek(0)
    magin = st.unpack('>4B', train_imagesfile.read(4))

    # Read the dimension of the Image data-set
    nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column

    # Reading the Image data
    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    return np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC)) 

# Get file paths
script_dir = os.path.dirname(__file__)
train_images_path = os.path.join(script_dir, "train-images-idx3-ubyte")
train_labels_path = os.path.join(script_dir, "train-labels-idx1-ubyte")
test_images_path = os.path.join(script_dir, "t10k-images-idx3-ubyte")
test_labels_path = os.path.join(script_dir, "t10k-labels-idx1-ubyte")

# Unpack files
train_images_array = unpack(train_images_path, train_labels_path)
test_images_array = unpack(test_images_path, test_labels_path)

# Save to .npy files
np.save('train_images_array', train_images_array)
np.save('test_images_array', test_images_array)