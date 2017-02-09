import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# First, load the image
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Print out its shape
'''
输出：(5528, 3685, 3)，表示这张图片高度5528像素，宽度2685像素，3种色深(R/G/B)
'''
print(image.shape)
'''
输出图片
'''
plt.imshow(image)
plt.show()