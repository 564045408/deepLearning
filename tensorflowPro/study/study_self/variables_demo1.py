'''
tf.constant 创建常量
tf.Variable 创建变量
tf.initialize_all_variables 初始化变量
session.run 运行会话
'''
#导入tensorflow模块，并调用tf
import tensorflow as tf
#创建一个名为x的常量，赋值35
x = tf.constant(35, name='x')
#创建一个名为y的变量，定义方程x+5
y = tf.Variable(x + 5, name='y')
#初始化变量
model = tf.initialize_all_variables()
#创建会话，在with结束后自动关闭会话
with tf.Session() as session:
	#运行创建的模型
    session.run(model)
	#只运行变量y，打印y的值
    print(session.run(y))