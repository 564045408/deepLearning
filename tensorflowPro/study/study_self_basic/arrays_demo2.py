import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 加载图片
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)
# 创建一个图片变量
x = tf.Variable(image, name='x')
#初始化
model = tf.initialize_all_variables()
#创建会话
with tf.Session() as session:
    #将图片矩阵的第一个维度(高)和第二个(宽)维度对调，第三个维度保持不变
    x = tf.transpose(x, perm=[1, 0, 2])
    #运行模型
    session.run(model)
    result = session.run(x)
#显示图片结果
plt.imshow(result)
plt.show()