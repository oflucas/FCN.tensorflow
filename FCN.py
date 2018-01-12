from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os, sys

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import read_KittiData as kitti_data
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ infer/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(2e5 + 1)
NUM_OF_CLASSES = 2 # MIT-SP 151
IMAGE_SIZE = None # MIT-SP 224


def vgg_net(weights, image, end_layer=None):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

        if end_layer is not None and end_layer == name:
            break

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        vgg_end_layer = 'conv4_4'
        image_net = vgg_net(weights, processed_image, end_layer=vgg_end_layer)
        conv_final_layer = image_net[vgg_end_layer]

        dropout = tf.nn.dropout(conv_final_layer, keep_prob=keep_prob)
        W_final = utils.weight_variable([1, 1, 512, NUM_OF_CLASSES], name="W_final")
        b_final = utils.bias_variable([NUM_OF_CLASSES], name="b_final")
        conv_final = utils.conv2d_basic(dropout, W_final, b_final)
        if FLAGS.debug:
            utils.add_activation_summary(conv_final)

        # now to upscale to actual image size
        deconv_shape2 = image_net["pool2"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, NUM_OF_CLASSES], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(conv_final, W_t2, b_t2, output_shape=tf.shape(image_net["pool2"]))
        fuse_2 = tf.add(conv_t2, image_net["pool2"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        W_t3 = utils.weight_variable([8, 8, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=4)

        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction", output_type=tf.int32)

    return tf.expand_dims(annotation_pred, axis=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") # [None, IMAGE_SIZE, IMAGE_SIZE, 3]
    annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)
    metric_ops = []
    acc = tf.contrib.metrics.accuracy(pred_annotation, annotation, name='ACC')
    tf.summary.scalar("ACC", acc)
    mean_iou, mean_iou_op = tf.metrics.mean_iou(annotation, pred_annotation, NUM_OF_CLASSES, name='mIOU')

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    if FLAGS.mode != 'infer':
        train_records, valid_records = kitti_data.read_dataset() #scene_parsing.read_dataset(FLAGS.data_dir)
        print(len(train_records))
        print(len(valid_records))
    else:
        test_records = kitti_data.test_data(FLAGS.data_dir)

    print("Setting up dataset reader")
    image_options = {'resize': True} 
    if FLAGS.mode == 'infer':
        image_options['infer'] = True
        test_dataset_reader = dataset.BatchDatset(test_records, image_options)
    else:
        if FLAGS.mode == 'train':
            train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...", ckpt.model_checkpoint_path)

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, ac = sess.run([loss, acc], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g, ACC: %g" % (datetime.datetime.now(), valid_loss, ac))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        time_elps = time.time()
        valid_loss, ac, pred, cf_mat, m_iou = sess.run([loss, acc, pred_annotation, mean_iou_op, mean_iou], 
                                         feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
        print ('iou:', cal_iou_by_cm(cf_mat), m_iou)
        m_iou = sess.run(mean_iou)
        time_elps = time.time() - time_elps
        print("%s ---> Validation_loss: %g, ACC: %g, mIOU: %g, time_elapsed: %gs" % (datetime.datetime.now(), valid_loss, ac, m_iou, time_elps))
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        pred_draw = valid_images.copy()
        pred_draw[:,:,:,1][pred > 0] = 255

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(itr))
            utils.save_image(pred_draw[itr].astype(np.uint8), FLAGS.logs_dir, name="draw_" + str(itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "infer":
        result_dir = os.path.join(FLAGS.data_dir, 'result')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        last_batch = False
        it = 0
        while (not last_batch):
            test_images, _, last_batch = test_dataset_reader.next_sequential_batch(FLAGS.batch_size)
            time_elps = time.time()
            pred = sess.run([pred_annotation], feed_dict={image: test_images, keep_probability: 1.0})
            time_elps = time.time() - time_elps
            print("batch no.%d, time_elapsed: %gs" % (it, time_elps))
            pred = np.squeeze(pred, axis=3)

            pred_draw = test_images.copy()
            pred_draw[:,:,:,1][pred > 0] = 255

            for itr in range(test_images.shape[0]):
                name_ = test_dataset_reader.files[it * FLAGS.batch_size + itr]['name'];
                utils.save_image(test_images[itr].astype(np.uint8), result_dir, name=name_)
                utils.save_image(pred_draw[itr].astype(np.uint8), result_dir, name=name_+'_I')

            it += 1


def cal_iou_by_cm(cf_mat):
    """
    input
    cf_mat: 2D np array, confusion matrix
    output
    ious: 1D np array, iou of each class
    """
    class_n = cf_mat.shape[0]
    true_pos = cf_mat[np.arange(class_n), np.arange(class_n)]
    total_truth = np.sum(cf_mat, axis=0)
    total_predict = np.sum(cf_mat, axis=1)
    denum = np.maximum(total_truth + total_predict - true_pos, 0.001).astype(np.float32);
    return true_pos / denum;
    

if __name__ == "__main__":
    tf.app.run()
