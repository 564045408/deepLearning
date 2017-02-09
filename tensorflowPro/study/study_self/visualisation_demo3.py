import tensorflow as tf
import numpy as np

# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([1.0, 2.0], name="w")
# Our model of y = a*x + b
#创建一个模型y'=ax+b，y'是预测的结果
y_model = tf.multiply(x, w[0]) + w[1]

# Our error is defined as the square of the differences
# error=实际结果-预测结果
error = tf.square(y - y_model)
# The Gradient Descent Optimizer does the heavy lifting
# 使用最速下降法将error值逐渐接近于0，接近速度0.01
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
# Normal TensorFlow - initialize values, create a session and run the model
# 初始化模型
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    #循环训练N次
    for i in range(8000):
        #x随机取值
        x_value = np.random.rand()
        #建立模型y=5x+6
        y_value = x_value * 5 + 6
        session.run(train_op, feed_dict={x: x_value, y: y_value})
        w_value = session.run(w)
        print(str(i)+":x="+str(x_value)+" y="+str(y_value)+" w0="+str(w_value[0])+
              " w1="+str(w_value[1])+" result="+str(x_value*w_value[0]+w_value[1])+
              " result_rea="+str(y_value)+" error="+str(y_value-x_value*w_value[0]-w_value[1]))

    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))