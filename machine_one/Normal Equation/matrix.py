#矩阵练习，技能要求如下:
'''
1)定义一个矩阵,形式为[[11,12,13,14,15],[21,22,23,……],……[51,……55]]。
2)对其进行切分
a.取前四列为矩阵X
b.取最后一列为矩阵Y
c.取X的前3/4行为矩阵trainX
d.取X的剩余行为矩阵testX
e.在trainX前添加一列，列值全部为1
'''
import numpy as np
data = np.arange(11, 61).reshape(5, -1) #56-11=45 45/5=9，由于一行需要10个 所以每行＋1 ，56+5=61
#arange首包尾不包
data = data[:, :5] #
#切分
x = data[:, :-1]#前四列为x
y = data[:, -1]#最后一列为y
m = len(x)
num = int(m*3/4)
trainx = x[:num, ] #从开头到num，行
testx = x[num:, ]  #从num到结束，行
tm = len(trainx) #trainx的行数
trainx = np.c_[np.ones((tm, 1)), trainx] #(tm, 1)tm行1列拼接与trainx
