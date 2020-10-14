from matplotlib import pyplot as plt
import numpy as np

def get_pixels(path, index):
    images_array = np.load(path)
    image = images_array[index]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    return pixels

def show_image(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()