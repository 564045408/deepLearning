import tensorflow as tf
import tensorflowvisu

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("data", one_hot=True)

sess = tf.InteractiveSession()
'''
创建占位符x、y_，
x代表输入的图片，是一个二维的张量，第一维表示数量不定的图片，第二维是一个展开成784(28x28)维的图片
y_表示输出的类别，也是一个二维的张量，每一行是一个十维的one-hot向量，对应这某一图片的标签类别（0-9）
'''
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
'''
创建变量W和b,使用tf.zero函数初始化变量为0
W表示图片中每一个像素的权重，为使W×x后获取N个10维的张量，所以W为一个[784,10]的向量
b表示每一个像素的偏移量
y_=W×x+b
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
'''
变量在session中初始化，初始值指定具体值（本例当中是全为零），
并将其分配给每个变量,可以一次性为所有变量完成此操作
'''
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
'''
定义模型
实现回归函数，tf.matmul是矩阵乘法运算，tf.softmax是求和运算∑
'''
y = tf.nn.softmax(tf.matmul(x,W) + b)
'''
定义损失函数
可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
y是预测的类别，y_是实际的类别，交叉熵cross_entropy=-∑y_×log(y)
tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了。我们计算的交叉熵是指整个minibatch的
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
'''
指定训练模型
我们已经定义好模型和训练用的损失函数，那么用TensorFlow进行训练就很简单了。因为TensorFlow知道整个计算图，
它可以使用自动微分法找到对于各个变量的损失的梯度值。TensorFlow有大量内置的优化算法 这个例子中，我们用最
速下降法让交叉熵下降，步长为0.01.
'''
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

datavis = tensorflowvisu.MnistDataVis()
'''
加载50次训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
循环1000次训练
在计算图中，你可以用feed_dict来替代任何张量，并不仅限于替换占位符。
'''
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/home/sy/PycharmProjects/deepLearning/temp', sess.graph)
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  #summary_str = train_step.run(summary_op)
  #summary_writer.add_summary(summary_str, i)

'''
评估训练模型的性能
tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是
模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用tf.equal来
检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#datavis.animate(train_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)