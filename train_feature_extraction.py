import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=832289)

print("Training set images: {}".format(X_train.shape))
print("Training set labels: {}".format(y_train.shape))
print("Validation set images: {}".format(X_valid.shape))
print("Validation set labels: {}".format(y_valid.shape))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.int32, shape=(None))
one_hot_y = tf.one_hot(y, 43)
resized = tf.image.resize_images(x, size=(227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], 43)
weight = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))
bias = tf.Variable(tf.zeros(shape=(43)))
logits = tf.add(tf.matmul(fc7, weight), bias)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operations = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_opeartions = optimizer.minimize(loss_operations)

# TODO: Train and evaluate the feature extraction model.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data, sess):
    n_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0

    for offset in range(0, n_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operations], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += accuracy * len(batch_x)
        total_loss += loss * len(batch_x)
    return total_accuracy / n_examples, total_loss / n_examples


EPOCHS = 1
BATCH_SIZE = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_examples = len(X_train)

    print("training for {} epochs...".format(EPOCHS))
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, n_examples, BATCH_SIZE):
            batch_x, batch_y = X_train[offset:offset + BATCH_SIZE], y_train[offset:offset + BATCH_SIZE]
            sess.run(training_opeartions, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy, validation_loss = evaluate(X_valid, y_valid, sess)
        print("EPOCHS {}: ... validation accuracy: {:.3f} loss: {:.3f}".format(i + 1, validation_accuracy, validation_loss))
        print()

    saver.save(sess, './alexnet')
    print("Model saved")
