import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
#cluster 聚集
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


#加载数据
x = np.loadtxt('test.txt')

meanx = np.mean(x, 0)
sigma= np.std(x, 0, ddof=1)
x = (x-meanx)/sigma

k = [] #聚类个数
jarr = []#代价函数

#肘部法则最优k
for i in range(20):
    model = KMeans(i+1) #随机生成聚类模型
    model.fit(x) #通过x配合训练
    k.append(i+1)#放入聚类个数
    jarr.append(model.inertia_)#放入对于代价函数

#需要用x轴的k指定位置，再用jarr配合
#代价曲线与对于散点
plt.plot(k, jarr)
#散点图需要有坐标生成点
plt.scatter(k, jarr)

for i in range(20):
    plt.annotate(str(i+1), xy=(k[i], jarr[i]))    
    
plt.title('肘部曲线')
plt.show()

model = KMeans(4)
model.fit(x)
plt.scatter(x[:, 0], x[:, 1], c=model.labels_, cmap=plt.cm.Paired)
plt.scatter(model.cluster_centers[:, 0], model.cluster_centers[:, 1],marker='x', s=60, c=['r','g','b','y'])
for i in range(k):
    plt.annotate('中心'+str(i+1), xy=(model.cluster_centers[i, 0], model.cluster_centers[i, 1]))
    
plt.title('聚类归属图')
plt.show()





    
    
    