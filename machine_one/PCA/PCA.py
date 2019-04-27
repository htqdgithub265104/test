import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets  as dts
from sklearn.decomposition import PCA

#加载数据
data = dts.load_breast_cancer()
x = data.data
y = data.target

#建模
model = PCA(n_components=2)  #components部件
z = model.fit_transform(x)#ttransform 改变
print('特征值方差', model.explained_variance_)
print('特征值方差比率', model.explained_variance_ratio_)

plt.scatter(z[y == 0, 0], z[y == 0, 1], c='y', label=data.target_names[0])
plt.scatter(z[y == 1, 0], z[y == 1, 1], c='b', label=data.target_names[1])
plt.legend()
plt.show()

#数据重建
newx = model.inverse_transform(z) #inverse
#benign(笔奶嗯)良性的  malignant(买累根特)恶性的