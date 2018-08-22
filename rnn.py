import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_step = 2000
batch_size = 40
keep_prob = tf.placeholder(tf.float32, [])

input_size = 28
timestep_size = 28
# layer_num = 4
class_num = 10

learning_rate = 0.001
data_dir = './input_data'

mnist = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(tf.float32, [None, input_size * timestep_size])
y = tf.placeholder(tf.int32, [None, class_num])

image = tf.reshape(x, [-1, timestep_size, input_size])

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=input_size, forget_bias=1.0, state_is_tuple=True)
rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
# mrnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * layer_num, state_is_tuple=True)

outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=image, initial_state=None, dtype=tf.float32,
                                         time_major=False)
output = tf.layers.dense(inputs=outputs[:, -1, :], units=class_num)

cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_predict = tf.equal(tf.argmax(y, axis=1), tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(max_step + 1):
        batch = mnist.train.next_batch(batch_size)
        if (i + 1) % 200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    print(
        "test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
    sess.close()
