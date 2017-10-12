import os.path
import time
import argparse
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pre-trained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob_tensor, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob_tensor, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def conv1x1_layer(input_layer, num_filters):
    return tf.layers.conv2d(input_layer,
                            num_filters,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


def deconv_layer(input_layer, num_filters, kernel, stride, name):
    return tf.layers.conv2d_transpose(input_layer,
                                      num_filters,
                                      kernel_size=(kernel, kernel),
                                      strides=(stride, stride),
                                      padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      name=name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer8 = deconv_layer(conv1x1_layer(vgg_layer7_out, num_classes), num_classes, 4, 2, 'layer8')
    layer8_4 = tf.add(layer8, conv1x1_layer(vgg_layer4_out, num_classes))

    layer9 = deconv_layer(layer8_4, num_classes, 4, 2, 'layer9')
    layer9_3 = tf.add(layer9, conv1x1_layer(vgg_layer3_out, num_classes))

    output = deconv_layer(layer9_3, num_classes, 16, 8, 'image_output')
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate_tensor, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate_tensor: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss, mean_iou)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    mean_iou = tf.metrics.mean_iou(tf.argmax(labels, 1), tf.argmax(logits, 1), num_classes)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_tensor).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss, mean_iou
tests.test_optimize(optimize)


def train_nn(sess, params, get_batches_fn, train_op, cross_entropy_loss, mean_iou,
             input_image, correct_label, keep_prob_tensor, learning_rate_tensor, writer):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param params: Training parameters
    :param get_batches_fn: Function to get batches of data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param mean_iou: TF Tensor for the interface over union
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob_tensor: TF Placeholder for dropout keep probability
    :param learning_rate_tensor: TF Placeholder for learning rate
    :param writer: TF Summary Writer to add training events
    """
    for epoch in range(1, params.epochs + 1):
        avg_train_loss = 0
        avg_iou = 0
        num_images = 0
        step = 0
        start = time.time()
        for images, labels in get_batches_fn(params.batch_size, params.augment_prob):
            _, loss, iou = sess.run([train_op, cross_entropy_loss, mean_iou],
                                    feed_dict={input_image: images,
                                               correct_label: labels,
                                               keep_prob_tensor: params.keep_prob,
                                               learning_rate_tensor: params.learning_rate})
            avg_train_loss += (loss * len(images))
            avg_iou += (iou[0] * len(images))
            num_images += len(images)
            step += 1

        end = time.time()
        elapsed_train_m, elapsed_train_s = [int(x) for x in divmod(end - start, 60)]
        avg_train_loss /= num_images
        avg_iou /= num_images
        epoch_summary = tf.Summary()
        epoch_summary.value.add(tag='loss', simple_value=avg_train_loss)
        epoch_summary.value.add(tag='iou', simple_value=avg_iou)
        writer.add_summary(epoch_summary, epoch)
        writer.flush()

        print("EPOCH {:>2d} Loss = {:10.8f} IoU = {:10.8f} Time = {:2d}m{:02d}s".format(
            epoch, avg_train_loss, avg_iou, elapsed_train_m, elapsed_train_s))
    return
tests.test_train_nn(train_nn)


def run():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Project")
    parser.add_argument('-e', dest='epochs', type=int, default=50, help="Number of epochs to train the model")
    parser.add_argument('-b', dest='batch_size', type=int, default=1, help="Training batch size")
    parser.add_argument('-l', dest='learning_rate', type=float, default=0.0001, help="Training learning rate")
    parser.add_argument('-k', dest='keep_prob', type=float, default=0.8, help="Training keep probability")
    parser.add_argument('-a', dest='augment_prob', type=float, default=0.0, help="Augment probability of ")
    params = parser.parse_args()

    num_classes = 2
    image_shape = (160, 576)

    data_dir = './data'
    runs_dir = './runs'
    logs_dir = './logs'
    models_dir = './models'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
    tf.set_random_seed(43)
    correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes))
    learning_rate_tensor = tf.placeholder(tf.float32, None)

    with tf.Session() as sess:
        session_timestamp = str(int(time.time()))

        runs_dir = os.path.join(runs_dir, session_timestamp)
        logs_dir = os.path.join(logs_dir, session_timestamp)
        models_dir = os.path.join(models_dir, session_timestamp)

        os.makedirs(runs_dir)
        os.makedirs(logs_dir)
        os.makedirs(models_dir)

        input_image, keep_prob_tensor, layer3out, layer4out, layer7out = load_vgg(sess, vgg_path)
        output = layers(layer3out, layer4out, layer7out, num_classes)
        logits, train_op, cross_entropy_loss, mean_iou = optimize(output,
                                                                  correct_label,
                                                                  learning_rate_tensor,
                                                                  num_classes)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logs_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_nn(sess,
                 params,
                 get_batches_fn,
                 train_op,
                 cross_entropy_loss,
                 mean_iou,
                 input_image,
                 correct_label,
                 keep_prob_tensor,
                 learning_rate_tensor,
                 writer)

        writer.close()

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_image)

        saver.save(sess, os.path.join(models_dir, 'fcn8'))

        # OPTIONAL: Apply the trained model to a video
    return


if __name__ == '__main__':
    run()
