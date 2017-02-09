import tensorflow as tf
#a=1+2--------------1.ADD1
a = tf.add(1, 2,name="add1")
#b=a×3=(1+2)×3------2.MUL1
b = tf.multiply(a, 3,name='mul1')
#c=4+5--------------3.ADD2
c = tf.add(4, 5,name="add2")
#d=c×6--------------4.MUL2
d = tf.multiply(c, 6,name='mul2')
#e=4×5--------------5.MUL3
e = tf.multiply(4, 5,name='mul3')
#f=c÷6--------------6.DIV1
f = tf.div(c, 6,name="div1")
#g=b+d--------------7.ADD3
g = tf.add(b, d,name="add3")
#h=b+d--------------8.MUL4
h = tf.multiply(g, f,name='mul4')

with tf.Session() as sess:
	#将session中的graph输出到output文件夹中，使用tensorboard查看
	writer = tf.train.SummaryWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()