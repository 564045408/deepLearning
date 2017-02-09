import tensorflow as tf
#创建一个N行3列的二维矩阵占位符x
x = tf.placeholder("float", [None, 3])
y = x * 2
with tf.Session() as session:
    #x矩阵的值如下
    x_data = [[1, 2, 3],
            [4, 5, 6],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)