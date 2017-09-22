from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
import urllib2
import  tensorflow.contrib.slim as slim
from nets import vgg
from preprocessing import vgg_preprocessing
from upsample_skimage import bilinear_upsample_weights,upsample_tf
# Load the mean pixel values and the function
# that performs the subtraction from each pixel
from preprocessing.vgg_preprocessing import (_mean_image_subtraction,
                                            _R_MEAN, _G_MEAN, _B_MEAN)

upsample_factor = 32
number_of_classes = 2
checkpoints_dir = '/home/dsz/checkpoints'
log_folder = '/home/dsz/PycharmProjects/log'
vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

IMAGE_HEIGHT = 416
IMAGE_WIDTH  = 416
Batch_size = 2
fig_size = [15, 4]
plt.rcParams["figure.figsize"] = fig_size




#image_filename = 'cat.jpg'
#annotation_filename = 'cat_annotation_2.png'

#annotation_filename = 'cat_annotation.png'
#
# image_filename = 'person.jpg'
# annotation_filename = 'person_annotation.png'

image_filename = ('cat.jpg','person.jpg')
annotation_filename = ('cat_annotation_2.png','person_annotation.png')


image_filename_placeholder = tf.placeholder(tf.string)
print("image_filename_placeholder=",image_filename_placeholder)
annotation_filename_placeholder = tf.placeholder(tf.string)
print("annotation_filename_placeholder=",annotation_filename_placeholder)

is_training_placeholder = tf.placeholder(tf.bool)

feed_dict_to_use = {image_filename_placeholder: image_filename,
                    annotation_filename_placeholder: annotation_filename,
                    is_training_placeholder: True}

#processed_images = tf.get_variable(name="processed_images",shape=[Batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,3],dtype=tf.float32)
#combined_masks = tf.get_variable(name="combined_masks",shape=[Batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,1],dtype=tf.float32)

processed_images_list =[]
combined_masks_list = []

for i in range(Batch_size):

    image_tensor = tf.read_file(image_filename_placeholder[i])
    annotation_tensor = tf.read_file(annotation_filename_placeholder[i])
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    annotation_tensor = tf.image.decode_png(annotation_tensor, channels=1)

    image_tensor = tf.image.resize_image_with_crop_or_pad(image=image_tensor,
                                                           target_height=IMAGE_HEIGHT,
                                                           target_width=IMAGE_WIDTH)

    annotation_tensor = tf.image.resize_image_with_crop_or_pad(image=annotation_tensor,

                                                                target_height=IMAGE_HEIGHT,
                                                                target_width=IMAGE_WIDTH)
    print("image_tensor=",image_tensor)
    print("annotation_tensor=",annotation_tensor)

    # Get ones for each class instead of a number -- we need that
    # for cross-entropy loss later on. Sometimes the groundtruth
    # masks have values other than 1 and 0.
    # one row represent a one hot code of  pixel's class
    class_labels_tensor = tf.equal(annotation_tensor, 255)
    print("class_labels_tensor",class_labels_tensor)

    background_labels_tensor = tf.not_equal(annotation_tensor, 255)
    print("background_labels_tensor",background_labels_tensor)

    # Convert the boolean values into floats -- so that
    # computations in cross-entropy loss is correct
    bit_mask_class = tf.to_float(class_labels_tensor)
    bit_mask_background = tf.to_float(background_labels_tensor)

    combined_mask = tf.concat(axis=2, values=[bit_mask_class,
                                                    bit_mask_background])

    print("combined_mask",combined_mask)

    combined_masks_list.append(combined_mask)

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image_tensor, name='ToFloat')

    # Subtract the mean pixel value from each pixel
    mean_centered_image = _mean_image_subtraction(image_float,
                                              [_R_MEAN, _G_MEAN, _B_MEAN])


    #temp_processed_image = tf.expand_dims(mean_centered_image, 0)
    processed_images_list.append(mean_centered_image)

combined_masks = tf.stack(combined_masks_list)
processed_images = tf.stack(processed_images_list)

print("combined_masks",combined_masks)
print ("processed_images",processed_images)

# Lets reshape our input so that it becomes suitable for
# tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
flat_labels = tf.reshape(tensor=combined_masks, shape=(-1, 2))
print("flat_labels",flat_labels)



#upsample_weight
upsample_filter_np = bilinear_upsample_weights(upsample_factor,
                                               number_of_classes)
print("upsample_filter_np",upsample_filter_np,upsample_filter_np.shape)

#Convert to a Tensor type
upsample_filter_tensor = tf.constant(upsample_filter_np)

# Define the model that we want to use -- specify to use only two classes at the last layer
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, end_points = vgg.vgg_16(processed_images,
                                    num_classes=2,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')
    print("logits",logits)

downsampled_logits_shape = tf.shape(logits)
print("downsampled_logits_shape=",downsampled_logits_shape)

# Calculate the ouput size of the upsampled tensor here only has a shape
upsampled_logits_shape = tf.stack([
                                  downsampled_logits_shape[0],
                                  downsampled_logits_shape[1] * upsample_factor,
                                  downsampled_logits_shape[2] * upsample_factor,
                                  downsampled_logits_shape[3]
                                 ])

print("upsampled_logits_shape=",upsampled_logits_shape)

# Perform the upsampling
upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor,
                                 output_shape=upsampled_logits_shape,
                                 strides=[1, upsample_factor, upsample_factor, 1])

print("upsampled_logits=",upsampled_logits)

# Flatten the predictions, so that we can compute cross-entropy for
# each pixel and get a sum of cross-entropies.
flat_logits = tf.reshape(tensor=upsampled_logits, shape=(-1, number_of_classes))


cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                          labels=flat_labels)

cross_entropy_sum = tf.reduce_sum(cross_entropies)


# Tensor to get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
pred = tf.argmax(upsampled_logits, dimension=3)
probabilities = tf.nn.softmax(upsampled_logits)

print("pred",pred)
print("probabilities",probabilities)

for i in range(Batch_size):
    pred_temp = pred[i,:,:]
    pred_temp_temp = tf.expand_dims(tf.expand_dims(pred_temp,2),0)
    tf.summary.image("pred"+str(i),tf.cast(pred_temp_temp,tf.float32))
    probabilities_temp = probabilities[i,:,:,:]
    probabilities_temp_0 = probabilities_temp[:,:,0]
    tf.summary.image("probabilities_temp_0"+str(i),tf.expand_dims(tf.expand_dims(probabilities_temp_0,2),0))
    probabilities_temp_1 = probabilities_temp[:,:,1]
    tf.summary.image("probabilities_temp_1"+str(i),tf.expand_dims(tf.expand_dims(probabilities_temp_1,2),0))

# Here we define an optimizer and put all the variables
# that will be created under a namespace of 'adam_vars'.
# This is done so that we can easily access them later.
# Those variables are used by adam optimizer and are not
# related to variables of the vgg model.

# We also retrieve gradient Tensors for each of our variables
# This way we can later visualize them in tensorboard.
# optimizer.compute_gradients and optimizer.apply_gradients
# is equivalent to running:
# train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_sum)


with tf.variable_scope("adam_vars"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    gradients = optimizer.compute_gradients(loss=cross_entropy_sum)

    for grad_var_pair in gradients:
        current_variable = grad_var_pair[1]
        current_gradient = grad_var_pair[0]

        # Relace some characters from the original variable name
        # tensorboard doesn't accept ':' symbol
        gradient_name_to_save = current_variable.name.replace(":", "_")

        # Let's get histogram of gradients for each layer and
        # visualize them later in tensorboard
        tf.summary.histogram(gradient_name_to_save, current_gradient)

    train_step = optimizer.apply_gradients(grads_and_vars=gradients)


# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.
vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])


# Here we get variables that belong to the last layer of network.
# As we saw, the number of classes that VGG was originally trained on
# is different from ours -- in our case it is only 2 classes.
vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

# Add summary op for the loss -- to be able to see it in
# tensorboard.
tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)



# Put all summary ops into one op. Produces string when
# you run it.
merged_summary_op = tf.summary.merge_all()


# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Create an OP that performs the initialization of
# values of variables to the values from VGG.
read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
                                   vgg_checkpoint_path,
                                   vgg_except_fc8_weights)


# Initializer for new fc8 weights -- for two classes.
vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

# Initializer for adam variables
optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)

with tf.Session() as sess:
    # Run the initializers.
    read_vgg_weights_except_fc8_func(sess)
    sess.run(vgg_fc8_weights_initializer)
    sess.run(optimization_variables_initializer)

    downsampled_logits_shape_r,upsampled_logits_shape_r = sess.run([downsampled_logits_shape,
                                                                    upsampled_logits_shape],
                                                                    feed_dict=feed_dict_to_use)

    print("downsampled_logits_shape_r",downsampled_logits_shape_r,downsampled_logits_shape_r.shape)

    print("upsampled_logits_shape_r",upsampled_logits_shape_r,upsampled_logits_shape_r.shape)

    # Let's perform 10 interations
    for i in range(10):
        loss, summary_string,train_step_r,pred_np,probabilities_np = sess.run([cross_entropy_sum,
                                                                              merged_summary_op,
                                                                              train_step,
                                                                              pred,
                                                                              probabilities  ],
                                                                              feed_dict=feed_dict_to_use)
        summary_string_writer.add_summary(summary_string, i)

        print("pred_np=",pred_np,pred_np.shape)
        print("probabilities_np=", probabilities_np, probabilities_np.shape)
        print("Current Loss: " + str(loss))

    feed_dict_to_use[is_training_placeholder] = False

    final_predictions, final_probabilities, final_loss = sess.run([pred,
                                                                   probabilities,
                                                                   cross_entropy_sum],
                                                                  feed_dict=feed_dict_to_use)
    print("final_predictions",final_predictions.shape)

    print("final_probabilities",final_probabilities.shape)

    print("Final Loss: " + str(final_loss))

summary_string_writer.close()



