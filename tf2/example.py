
import tensorflow as tf
from utils import load_pk

batch_size = 4

weights = load_pk('weights.pk')
data = load_pk('data.pk')
labels = load_pk('labels.pk')


data = tf.reshape(tf.convert_to_tensor(data[:batch_size, :]), (batch_size, -1))
labels = tf.reshape(tf.convert_to_tensor(labels[:batch_size]), (-1,))

tf.random.set_seed(127)

# build the model
weights = tf.Variable(weights)
logits = data @ weights
loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))

accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.cast(labels, dtype=tf.int32),
                       tf.argmax(logits, axis=1, output_type=tf.int32)),
              dtype=tf.float32))

print(accuracy)
print(loss)




# compute the covariances


