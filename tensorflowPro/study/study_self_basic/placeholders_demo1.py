import tensorflow as tf
#创建一个占位3的占位符x
x = tf.placeholder("float", 3)
y = x * 2

with tf.Session() as session:
    #x的值为[1,2,3]
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)