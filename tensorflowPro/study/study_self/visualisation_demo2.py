import tensorflow as tf
with tf.name_scope("MyOperationGroup"):
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="Add_these_numbers")
        b = tf.multiply(a, 3)
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="And_These_ones")
        d = tf.multiply(c, 6, name="Multiply_these_numbers")

with tf.name_scope("Scope_C"):
    e = tf.multiply(4, 5, name="B_add")
    f = tf.div(c, 6, name="B_mul")
g = tf.add(b, d)
h = tf.multiply(g, f)

with tf.Session() as sess:
	#将session中的graph输出到output文件夹中，使用tensorboard查看
	writer = tf.train.SummaryWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()