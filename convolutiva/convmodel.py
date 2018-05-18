# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h

num_classes = 3
batch_size = 4
# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity=min_after_dequeue + 3 * batch_size

    example_batch_list = [] #almacena muestras
    label_batch_list = [] #almacena etiquetas

    for i, p in enumerate(paths):
        filename= tf.train.match_filenames_once(p)
        filename_queue= tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        #_ es una variable que no utilizaremos mas
        _, file_image= reader.read(filename_queue)
        #image --> imagen decodificada" \
        #label -- etiqueta"
        image, label = tf.image.decode_jpeg(file_image),  one_hot(int(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image=tf.image.rgb_to_grayscale(image)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------
def myModel(X,reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        #o5 = tf.layers.conv2d(inputs=o4, filters=64, kernel_size=3, activation=tf.nn.relu)
        #o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)


        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 33 * 18 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["data3/cap/*.jpg", "data3/paper/*.jpg", "data3/toy/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data3/valid/cap/*.jpg", "data3/valid/paper/*.jpg", "data3/valid/toy/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data3/test/cap/*.jpg", "data3/test/paper/*.jpg", "data3/test/toy/*.jpg"], batch_size=batch_size)

print (batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_train = []
    error_valid = []


    """*****Entrenamiento****"""
    for _ in range(200):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Epoch:", _, "---------------------------------------------")
            error_train.append(sess.run(cost))
            error_valid.append(sess.run(cost_valid))
            print ("Trainning Error:", sess.run(cost))
            print("Validation Error:", sess.run(cost_valid))
    total = 0.0
    error = 0.0
    acierto=0.0
    test_data = sess.run(label_batch_test)
    test_hoped = sess.run(example_batch_test_predicted)

    for real_data, hoped in zip(test_data, test_hoped):
        if np.argmax(real_data) != np.argmax(hoped):
            error += 1
        else:
            acierto+=1
        total += 1
    print error
    print acierto
    fallo = error / total * 100
    acierto=acierto/total * 100
    print("El porcentaje de error es: ", fallo, "% y el de exito ", acierto, "%")
    tr_handle, = plot.plot(error_train)
    #plot.show()
    vl_handle, = plot.plot(error_valid)
    #plot.show()


    plot.legend(handles=[tr_handle, vl_handle],
                labels=['Error entrenamiento', 'Error validacion'])
    plot.title("Learning rate = 0.01")
    plot.show()
    plot.savefig('Grafica_entrenamiento.png')

    #coord.request_stop()
    #coord.join(threads)
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

