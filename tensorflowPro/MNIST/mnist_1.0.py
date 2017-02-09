import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("data", one_hot=True)
#创建28x28的平面向量，每张图由784维向量组成，一共由N张图
x = tf.placeholder(tf.float32, [None, 784])
#创建一个权值W，使之与每张图的向量相乘得一个十维的向量，这个向量表示这张图所表示的数字的概率
W = tf.Variable(tf.zeros([784, 10]))
#创建一个偏移量b，使之与之前得出的十维向量相加
b = tf.Variable(tf.zeros([10]))
#结果得出y=x×W+b，然后进行线性回归函数，softmax是一个回归模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
#创建新的10维向量placeholder
y_ = tf.placeholder("float", [None,10])
#reduce_sum是求和函数，y_是实际分布，y是预测的分布，交叉熵cross_entropy=-∑y'log(y)
#使用tf.log计算y中每个元素的对数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))