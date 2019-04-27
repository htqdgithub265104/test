#神经网络调库(手写数字识别)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics  import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
'''
sklearn包含了很多机器学习的方法
neural_network 神经网络 牛若_奈特喔渴
confusion_matrix  混淆矩阵  啃飞神_梅揣渴丝
metrix 买垂渴丝
classification_report 分类 class 佛AK神 报告 
MPL machine(莫什)  program language(懒哥瑞之)   机器程序语言
'''
#加载数据
x = np.loadtxt('imgX.txt', delimiter=',')
y = np.loadtxt('labely.txt', delimiter=',')
# x(5000, 400)  y(5000,)
#洗牌
np.random.seed(0)
m, n = x.shape
order = np.random.permutation(m) #random(软的木) 随机 permutaion(普绕米他神) 置换
x = x[order]
y = y[order]

#归一化特征缩放
maxx = np.max(x)
minx = np.min(x)
x = (x-minx)/(minx-maxx)

#将标签所以为10的改为0
y[y == 10] = 0

#切分
m = x.shape[0]
num = int(m*0.7)
trainx, testx = np.split(x, [num])
trainy, testy = np.split(y, [num])

#建模
model = MLPClassifier(max_iter=150)#建立多层感知机分类器
model.fit(trainx, trainy)#训练模型
testh = model.predict(testx)#预测数据

#显示混淆矩阵和分类报告
print('混淆矩阵:\n', confusion_matrix(testy, testh))
print('分类报告:\n', classification_report(testy, testh))
print('精度', model.score(testx, testy))

#显示图片
row = int(np.sqrt(n)) #根据特征得到图片行列数 n=400， 20*20=40  一列表示一个特征，所以开根号之后，让其x*y=400显示图片
img = testx[2].reshape(row, row) #将图片整形成20*20
plt.imshow(img)
plt.show()
print('预测值:', testh[2])
print('真实值:', testy[2])
