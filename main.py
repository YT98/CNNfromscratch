import numpy as np

import image_utils as imu
from Convolution import Convolution

image = imu.get_pixels('train_images_array.npy', 5)

def main():
    conv = Convolution(3, (2, 2))
    conv_image = conv.forward_propagation(image)
    imu.show_image(image)
    imu.show_image(conv_image[0])

if __name__ == "__main__":
    main()