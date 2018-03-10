import tensorflow as tf
import pandas as pd
import numpy as np
import os


def load_iris(path):
    df = pd.read_csv(path)
    df = df.reindex(np.random.permutation(df.index))  # shuffle data

    headers = df.columns.values

    def normalize(v):
        return (v - np.mean(v)) / (np.max(v) - np.min(v))

    normalized_df = df[headers[:4]].apply(normalize)

    values = normalized_df.as_matrix()
    output = pd.get_dummies(df[headers[4]]).as_matrix()

    return values, output

if __name__ == '__main__':
    values, output = load_iris('./iris.csv')
    size = 150
    ratio = 0.7
    train_size = int(size * ratio)
    test_size = size - train_size

    train_x = values[:train_size, :].reshape((train_size, 4))
    train_y = output[:train_size].reshape((train_size, 3))
    test_x = values[train_size:, :].reshape((test_size, 4))
    test_y = output[train_size:].reshape((test_size, 3))

    x = tf.placeholder(tf.float32, shape=(None, 4))
    y = tf.placeholder(tf.float32, shape=(None, 3))
    W = tf.Variable(tf.random_uniform([4, 3]))
    b = tf.Variable(tf.random_uniform([1, 3]))

    model = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = -tf.reduce_sum(y * tf.log(model))
    optimize = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(optimize, feed_dict={x: train_x, y: train_y})

        print('AVG Loss: ', sess.run(cost, feed_dict={x: test_x, y: test_y})/test_size)

        for i in range(test_size):
            print(sess.run(tf.argmax(model, axis=1), feed_dict={x: test_x[i].reshape((1, 4))}), end=' -> ')
            print(np.argmax(test_y[i]))
