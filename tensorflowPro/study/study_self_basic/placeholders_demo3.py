import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
'''
功能：从一张完整的图片中截取部分图片
'''
# First, load the image again
filename = "MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [1000, 0, 0], [3000, 1000, -1])
'''
1，函数原型 tf.slice(inputs,begin,size,name='')

2，用途：从inputs中抽取部分内容

     inputs：可以是list,array,tensor

     begin：n维列表，begin[i] 表示从inputs中第i维抽取数据时，相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据

     size：n维列表，size[i]表示要抽取的第i维元素的数目

     有几个关系式如下:

         （1） i in [0,n]

         （2）tf.shape(inputs)[0]=len(begin)=len(size)

         （3）begin[i]>=0   抽取第i维元素的起始位置要大于等于0

         （4）begin[i]+size[i]<=tf.shape(inputs)[i]
上述函数表示从第一个维度的第1000个元素，第二三个维度第0个元素开始，第一个维度抽取3000个元素，第二三个维度从第0个元素（begin）剩余的元素都要被抽取
'''
with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()