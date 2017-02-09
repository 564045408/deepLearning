#导入科学计算包
from numpy import *
#导入运算符模块
import operator
#创建数据集和标签
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group , labels