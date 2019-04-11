#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
from datetime import timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, 
        layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    # Load VGG model from saved file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # Extract the layers we need to create our new network
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Define kernel initializer and regularizer
    init = tf.truncated_normal_initializer(stddev = 0.01)
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    # Perform 1x1 convolutions on layer 3, 4 and 7 with L2 regularizer for the weights
    conv_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                   kernel_initializer=init, kernel_regularizer=reg)
    conv_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                   kernel_initializer=init, kernel_regularizer=reg)
    conv_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                   kernel_initializer=init, kernel_regularizer=reg)
    # Perform first transposed convolution from layer 7
    deconv_1 = tf.layers.conv2d_transpose(conv_layer7, num_classes, 4, 2, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)
    # Add the first skip connection from layer 4
    skip_1 = tf.add(deconv_1, conv_layer4)
    # Perform second transposed convolution on first skip
    deconv_2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)
    # Add the second convolution to layer 3
    skip_2 = tf.add(deconv_2, conv_layer3)
    # Perform third transposed convolution and match input image size
    layer_output = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)
    return layer_output;
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Logits is a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # Create output labels and reshape to match size
    labels = tf.reshape(correct_label, (-1, num_classes))
    # Use standard cross-entropy-loss as loss function
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # For the optimizer, use Adam
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
             input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  
        Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    print("Training...")
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    # Go through all epochs
    for epoch in range(epochs):
        loss = None
        s_time = time.time()
        # Go through all batches
        for image, labels in get_batches_fn(batch_size):
            # Train model and get loss
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image: image,
                           correct_label: labels,
                           keep_prob: .8,
                           learning_rate: 1e-4})
        # Print loss for each epoch
        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch + 1, 
              epochs, loss, str(timedelta(seconds=(time.time() - s_time)))))
tests.test_train_nn(train_nn)

def run():
    # Configuration
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    # Hyper-Parameter
    batch_size = 10
    epochs = 20
    learning_rate = tf.constant(1e-4)
    # Run tensorflow session
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 
                                                   'data_road/training'), image_shape)
        # Load pretrained VGG Model into TensorFlow and extract layer
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
            load_vgg(sess, vgg_path)
        # Create FCN model
        layer_output = layers(vgg_layer3_out, vgg_layer4_out, 
                              vgg_layer7_out, num_classes)
        # Build the TensorFlow loss and optimizer operations
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0],
                                                    image_shape[1], num_classes])
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label,
                                                        learning_rate, num_classes)
        # Train model using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
                 input_image, correct_label, keep_prob, learning_rate)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, 
                                      logits, keep_prob, input_image)
# Main Program
if __name__ == '__main__':
    run()
