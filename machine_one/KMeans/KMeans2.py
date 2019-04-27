import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#KMeans为聚类算法，不需要标签，且不需要切分
x = np.loadtxt('test.txt')

#特征缩放
meanx = np.mean(x, 0)
sigma = np.std(x, 0, ddof=1)
x = (x-meanx)/sigma

#肘部曲线最优k
k = np.arange(1, 21)
jarr = []
for i in k:
    model = KMeans(i) #不要用i+1
    model.fit(x)
    jarr.append(model.inertia_)
    plt.annotate(str(i), xy=(i, model.inertia_)) #显示数值

plt.plot(k, jarr) #代价曲线
plt.scatter(k, jarr) #显示散点图
plt.show()

k = 4
model = KMeans(k)
carr = model.fit_predict(x) #训练并预测样本归属
muarr = model.cluster_centers_ #获得聚类中心

#画样本散点图
plt.scatter(x[:, 0], x[:, 1], c=carr, cmap=plt.cm.Paired) #确定归属
plt.scatter(muarr[:, 0], muarr[:, 1], s=100, c=['r', 'g', 'b', 'y'], marker='^')
#特征分为四份，muarr也是四份

for i in range(k):
    plt.annotate('中心'+str(i+1), xy=(muarr[i, 0], muarr[i, 1]))
#使用下标获取每个聚类中心，annotate中没有散点图的全部读取功能

plt.title('聚类归属散点图')
plt.show()

















