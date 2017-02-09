import tensorflow as tf
import resource
import numpy as np
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
X = tf.constant(np.eye(1000))
Y = tf.constant(np.random.randn(1000, 300))
session = tf.InteractiveSession()
Z = tf.matmul(X, Y)
print(Z.eval())
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
session.close()