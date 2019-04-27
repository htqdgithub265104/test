'''
技能要求:
用sklearn库对irir(鸢yuan尾花)数据集进行分类，
取data中y=[1,2]的两个类，用特征中的x2,x3来分类

SVM支持向量机，通过支持向量，最大间距分类数据也称为最大间距分类器

调库:数据整齐，需要洗牌，不需要初始化
'''
#1.导库
import numpy as np
import matplotlib.pyplot as plt #一种绘图工具
import matplotlib as mpl  
import sklearn.datasets as dts
from sklearn.svm import SVC #svm支持向量机  support  vector machine
from pylab import * #中文显示两种方式
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False #现在正负号

#加载数据 
data = dts.load_iris()
x = data['data'] #data.data
y = data['target'] #data.target
#选特征，选取y=[1, 2] x2 x3的
x = x[y!=0, 1:3]
y = y[y!=0]

#特征缩放
'''
归一化缩放
maxx = np.max(x, 0)
minx = np.min(x, 0)
x = (x-minx)/maxx-minx
'''

#标准缩放
meanx = np.mean(x, 0)
sigma = np.std(x, 0, ddof=1)
x = (x-meanx)/sigma

#洗牌
m = len(x)
np.random.seed(0)
order = np.random.permutation(m) #permutation 排列
x = x[order]
y = y[order]

#切分
num = int(0.7*m)
trainx, testx = np.split(x, [num])
trainy, testy = np.split(y, [num])

#训练
model = SVC(C=50,gamma=0.1, kernel='rbf')
#C表示惩罚力度(线的位置由影响)，gamma大欠拟合，小过拟合，c大过拟合，c欠拟合 kennerl要点
model.fit(trainx, trainy)
score = model.score(testx,testy)
print('测试集精度', score)
testh = model.predict(testx)

#输出
print('每个类别支持向量个数', model.n_support_)
print('支持向量索引', model.support_)
print('支持向量', model.support_vectors_)
support = model.support_vectors_ #支持向量

#画样本和支持向量机散点图
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired,edgecolors='k',s=40, zorder=4)
#c=y根据y按类别自动给颜色，cmap自动匹配一对颜色，edgecolor点边线的颜色，zorder让点在面的上面 s表示元素大学
plt.scatter(support[:, 0], support[:, 1], facecolors='none', edgecolors='k', s=80, zorder=4)
#facecolors让样本点面的颜色时透明的

#获取网格最小值最大值，规定范围
minx1, maxx1, minx2, maxx2 = min(x[:, 0]), max(x[:, 0]), min(x[:, 1]), max(x[:, 1])

#生成200行，200列的格网 mgrid制作网格
xx, yy = np.mgrid[minx1:maxx1:200j, minx2:maxx2:200j]


#计算格网点到超平面的距离 decision 决定 function功能  ravel散开，计算(xx,yy)的距离,从而分类，
z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])#

#将z整形成xx矩阵
z = z.reshape(xx.shape)  

#画分界线 contourf 填色等位线
plt.contourf(xx, yy, z>0, cmap=plt.cm.Paired) #填充类别颜色 
#画分界线
plt.contour(xx, yy, z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors=['r','k', 'b'])
#levels规定线的划为，linestyles规定线的形状，colors规定线的颜色
plt.title('方差=%.2f,精度=%.2f'% (np.var(testh),score))
plt.show()













