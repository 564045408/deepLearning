import tensorflow as tf
#创建一个常数矩阵
x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')
model = tf.initialize_all_variables()
with tf.Session() as session:
	session.run(model)
	print(session.run(y))
