from preprocessing import vgg_preprocessing
from preprocessing.vgg_preprocessing import (_mean_image_subtraction,_R_MEAN,_G_MEAN,_B_MEAN)
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib2
import tensorflow.contrib.slim as slim
from datasets import imagenet
from nets import vgg
checkpoints_dir = '/home/dsz/checkpoints'


# Function to nicely print segmentation results with
# colorbar showing class names
def discrete_matshow(data, labels_names=[], title=""):

    print ("range",np.max(data) - np.min(data) + 1 )
    print("labes count",np.array(labels_names).shape)

    # get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)
    #cmap = plt.get_cmap('Dark2', np.max(data) - np.min(data) + 1)

    print ("cmap=", cmap)

    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin=np.min(data) - .5,
                      vmax=np.max(data) + .5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat,ticks=np.arange(np.min(data), np.max(data) + 1))

    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')


with tf.Graph().as_default():
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")

    image_string = urllib2.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image, name='ToFloat')
    print("image_float",image_float)

    # Subtract the mean pixel value from each pixel
    processed_image = _mean_image_subtraction(image_float,
                                              [_R_MEAN, _G_MEAN, _B_MEAN])
    print("processed_image", processed_image)

    input_image = tf.expand_dims(processed_image, 0)
    print("input_image", input_image)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        # spatial_squeeze option enables to use network in a fully
        # convolutional manner
        logits, _ = vgg.vgg_16(input_image,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=False)
        print("logits", logits)

    # For each pixel we get predictions for each class
    # out of 1000. We need to pick the one with the highest
    # probability. To be more precise, these are not probabilities,
    # because we didn't apply softmax. But if we pick a class
    # with the highest value it will be equivalent to picking
    # the highest value after applying softmax
    pred = tf.argmax(logits, dimension=3)
    print("pred", pred)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        init_fn(sess)
        segmentation, np_image = sess.run([pred, image])

# Remove the first empty dimension
print("segmentation", segmentation.shape)
segmentation = np.squeeze(segmentation)
print("segmentation", segmentation,segmentation.shape)

# Let's get unique predicted classes (from 0 to 1000) and
# relable the original predictions so that classes are
# numerated starting from zero
unique_classes, relabeled_image = np.unique(segmentation,
                                            return_inverse=True)
print("unique_classes",unique_classes, unique_classes.shape)
print("relabeled_image", relabeled_image,relabeled_image.shape)


segmentation_size = segmentation.shape

names = imagenet.create_readable_names_for_imagenet_labels()

relabeled_image = relabeled_image.reshape(segmentation_size)
print("relabeled_image", relabeled_image,relabeled_image.shape)

labels_names = []

for index, current_class_number in enumerate(unique_classes):
    labels_names.append(str(index) + ' ' + names[current_class_number + 1])

discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
plt.figure()
plt.imshow(np_image)
plt.show()