from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib2
import tensorflow.contrib.slim as slim
from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

checkpoints_dir = '/home/dsz/checkpoints'


image_size = vgg.vgg_16.default_image_size
print("image_size",image_size) #224


with tf.Graph().as_default():
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")
    image_string = urllib2.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    print("image", image)

    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    print("processed_image", processed_image)

    processed_images = tf.expand_dims(processed_image, 0)
    print("processed_images", processed_images)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=True)
        print("logits", logits)

        probabilities = tf.nn.softmax(logits)

        print("probabilities", probabilities)

        print(os.path.join(checkpoints_dir, 'vgg_16.ckpt'))

        print( slim.get_model_variables('vgg_16') )

        init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'),slim.get_model_variables('vgg_16'))
        print ("---------------------------------------")
    with tf.Session() as sess:
        init_fn(sess)
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        print("probabilities", probabilities,probabilities.shape)

        probabilities = probabilities[0, 0:]
        print("probabilities", probabilities,probabilities.shape)

        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x: x[1])]

    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    plt.imshow(network_input)
    plt.suptitle("network_input image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    plt.imshow(network_input / (network_input.max() - network_input.min()))
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index + 1]))

        res = slim.get_model_variables()
