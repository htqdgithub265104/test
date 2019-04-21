import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
#sigmoid函数
def g(z):
    h = 1.0/(1+np.exp(-z))
    return h
#创建模型
def model(x, theta):
    z = x.dot(theta)
    h = g(z)
    return h
#代价函数
def costfunc(h, y, R):
    m = len(h)
    J = -1.0/m*np.sum(y*np.log(h)+(1-y)*np.log(1-h))+R
    return J
#梯度下降最优解
def gradDesc(x, y, alpha=0.01, max_iter=15000, lamada=1.2):
    m, n = x.shape
    theta = np.zeros((n, 1))
    jarr = np.zeros(max_iter)
    for i in range(max_iter):
        h = model(x, theta)
        theta_r = theta.copy()
        theta_r[0] = 0
        R = lamada/(2*m)*np.sum(np.square(theta_r))
        jarr[i] = costfunc(h, y, R)
        e = h - y
        deltatheta = 1.0/m*(x.T.dot(e)+alpha*lamada)
        theta -= alpha*deltatheta
    return jarr, theta
#精度
def accuracy(h, y):
    m = len(h)
    count = 0
    for i in range(m):
        h[i] = np.where(h[i] >= 0.5, 1, 0)
        if h[i] == y[i]:
            count += 1
    return count/m
#画图
def draw(x, y, theta):
    ones = y[:, 0] == 1
    zeros = y[:, 0] == 0
    plt.scatter(x[ones, 2], x[ones, 3], c='r', label='正向类')
    plt.scatter(x[zeros, 2], x[zeros, 3], c='b', label='负向类')

    minx1 = x[:, 1].min()
    maxx1 = x[:, 1].max()
    minx2 = x[:, 2].min()
    maxx2 = x[:, 2].max()
    minx3 = -(theta[0]+theta[1]*minx1+theta[2]*minx2)/theta[3]
    maxx3 = -(theta[0]+theta[1]*maxx1+theta[2]*maxx2)/theta[3]
    plt.plot([minx2, maxx2], [minx3, maxx3])
    plt.title('模型精度 %.2f' % accuracy(theta, y))
    plt.legend()
    plt.show()
#预处理
def preprocess(x, y):
    m = len(x)
    meanx = np.mean(x, 0)
    sigma = np.std(x, 0, ddof=1)
    x = (x-meanx)/sigma
    x = np.c_[np.ones((m, 1)), x]
    y = np.c_[y]
    return x, y
#加载数据
data = np.loadtxt('Logistic.txt', delimiter=',')
x, y = np.split(data, [-1], axis=1)
m = len(x)
np.random.seed(0)
order = np.random.permutation(m)
x = x[order]
y = y[order]

m = len(x)
num = int(m*0.7)
trainx, testx = np.split(x, [num])
trainy, testy = np.split(y, [num])

trainx, trainy = preprocess(trainx, trainy)
testx, testy = preprocess(testx, testy)

jarr, theta = gradDesc(trainx, trainy)
testh = model(testx, theta)
trainh = model(trainx, theta)

draw(testx, testy, testh)
draw(trainx, trainy, trainh)

print('测试集精度', accuracy(testh, testy))
print('训练集精度', accuracy(trainh, trainy))
#代价曲线
plt.plot(jarr)
plt.title('代价曲线')
plt.show()

#sigmoid函数图
a = np.arange(-10, 10)
b = g(a)
plt.title('sigmoid函数图')
plt.plot(a, b)
plt.show()












