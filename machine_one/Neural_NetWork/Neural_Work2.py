'''
用神经网络模型对西瓜数据进行训练，并对训练集进行预测
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#sigmoid函数
def g(z, deriv=False): #deriv判断是否对函数求导
    if deriv == True:
        return z*(1-z)
    return 1.0/(1+np.exp(-z))

#模型
def model(x, theta1, theta2):
    #求第二层特征
    z2 = x.dot(theta1)
    a2 = g(z2)
    #第三层特征,为最终预测值
    z3 = a2.dot(theta2)
    a3 = g(z3)
    return a2, a3

#计算代价
def costfunc(h, y):
    m = len(h)
    e = h-y
    J = -1.0/m*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
    # J = -1.0/m(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))
    return J
#反向转播 BP算法
def BP(a1, a2, a3, y, theta1, theta2, alpha):
    m = len(a1)
    delta3 = a3-y
    delta2 = delta3.dot(theta2.T)*g(a2, deriv=True)
    deltatheta2 = 1.0/m*a2.T.dot(delta3)
    deltatheta1 = 1.0/m*a1.T.dot(delta2)
    theta1 -= alpha*deltatheta1
    theta2 -= alpha*deltatheta2
    return theta1, theta2
def gradDesc(x, y, alpha=0.1, max_iter=1500):
    m, n = x.shape
    np.random.seed(0)
    theta1 = 2*np.random.rand(n, 17)-1 #第一次theta定义shape，避免梯度爆炸，范围0-2，1- -1(值有正有负)
    theta2 = 2*np.random.rand(17, 1)-1
    jarr = np.zeros(max_iter)
    for i in range(max_iter):
        a2, a3 = model(x, theta1, theta2)
        jarr[i] = costfunc(a3, y)
        theta1, theta2 = BP(x, a2, a3, y, theta1, theta2, alpha)
    return jarr, theta1, theta2
#计算精度
def score(h, y):
    m = len(h)
    count = 0
    for i in range(m):
        h[i] = np.where(h[i] >= 0.5, 1, 0)
        if h[i] == y[i]:
            count += 1
    return count/m

#预处理
def preprocess(x, y):
    #归一化
    maxx = np.max(x, axis=0)
    minx = np.min(x, 0)
    x = (x-minx)/(maxx-minx)
    m = len(x)
    x = np.c_[np.ones((m, 1)), x]
    y = np.c_[y]

    return x, y

X1 = [0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719]
X2 = [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]
Y = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]

x = np.c_[X1, X2]
y = np.c_[Y]
m = len(x)
#洗牌
np.random.seed(0)
order = np.random.permutation(m)
x = x[order]
y = y[order]
#切分
num = int(m*0.7)
trainx, testx = np.split(x, [num])
trainy, testy = np.split(y, [num])
trainx, trainy = preprocess(trainx, trainy)
testx, testy = preprocess(testx, testy)

#训练模型
jarr, theta1, theta2 = gradDesc(trainx, trainy)
a2, testh = model(testx, theta1, theta2)
a2, trainh = model(trainx, theta1, theta2)
print('精度:', score(testh, testy))
print('精度:', score(trainh, trainy))
#测试集精度过低，表示欠拟合，特征过于简单，增加复杂的
plt.plot(jarr)
plt.show()



