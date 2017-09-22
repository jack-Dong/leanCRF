from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib2
import tensorflow.contrib.slim as slim
from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

image_size = 248



image = plt.imread("cat.jpg")

processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
print("processed_image",processed_image)

with tf.Session() as ss:

    plt.imshow(image)
    #plt.show()
    plt.figure()

    processed_imager = ss.run([processed_image])
    processed_imager = np.array(processed_imager)
    processed_imager = processed_imager[0,:,:,:]
    print("processed_imager",processed_imager,processed_imager.shape)
    plt.imshow(processed_imager/(processed_imager.max() - processed_imager.min()) )
    plt.show()


