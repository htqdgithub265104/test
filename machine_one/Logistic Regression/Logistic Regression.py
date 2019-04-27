'''
要求:
1.逻辑回归模型
2.代价
3.梯度下降函数
4.正则化项
5.计算模型精度 (通过x使用模型得出预测值)
6.特征画图,分界线
'''
#1.导入库函数
import numpy as np #from numpy as *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']# 正确显示中文(如:title)
# sans:无(塞恩s) serif:衬线 (赛偌儿夫)
# 代码二：from pylab import *
mpl.rcParams['axes.unicode_minus'] = False #正确显示负数
#axes:坐标轴(阿渴谁丝)  unicode:采用双字节(1字节8位,一中文来两字节)对字符进行编码(油你扣的) minus:负数(迈纳斯)

#sigmoid函数
def g(z):
    h = 1.0/(1+np.exp(-z))
    return h

#创建模型
def model(x ,theta):
    z = x.dot(theta)
    h = g(z)
    return h
#代价函数
def costfunc(h, y, lamda, theta):
    m = len(h)

    '''
    theta_r = theta.copy()
    theta_r[0] = 0
    '''

    #正则化两种方式
    R = lamda/(2.0*m)*np.sum(theta.T.dot(theta)) #theta_r
    # R = lamda/(2.0*m)*np.sum(np.square(theta)) #theta_r

    # 代价函数两种方式
    J = -1.0/m*np.sum(y*np.log(h)+(1-y)*np.log(1-h))+R
    #J = -1.0/m(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))+R
    return J

#定义梯度下降算法
def gradDesc(x, y, alpha=0.01, max_iter=1500, lamda=10): #lamda解决过拟合

    m, n = x.shape #样本个数m， 特征个数
    theta = np.zeros((n, 1)) #初始化theta
    jarr = np.zeros(max_iter)

    #开始梯度下降
    for i in range(max_iter):
        h = model(x, theta)#随机生成预测值
        jarr[i] = costfunc(h, y, lamda, theta) #计算代价值
        e = h-y
        deltatheta = 1.0/m*(np.dot(x.T, e)+lamda/m*theta)
        #计算deltatheta  lamda/m*theta是正规化项的梯度下降公式
        theta -= alpha*deltatheta #更新theta

    return jarr, theta
#计算预测精度
def accuracy(h, y):
    m = len(h)
    count = 0
    for i in range(m):
        h[i] = np.where(h[i] >= 0.5, 1, 0)
        if h[i] == y[i]:
            count+=1
    return count/m

def draw(x, y, theta):
    plt.scatter(x[y[:, 0] == 1, 1], x[y[:, 0] == 1, 2], c='b', label='正向类')
    plt.scatter(x[y[:, 0] == 0, 1], x[y[:, 0] == 0, 2], c='r', label='负向类')
    
    minx1 = np.min(x[:, 1])
    maxx1 = np.max(x[:, 1])

    '''
    minx1 = x[:,1].min()
    maxx1 = x[:,1].max()
    '''
    
    #找到横轴纵轴两个点，得到分界线
    minx2 = -(theta[0]+theta[1]*minx1)/theta[2]
    maxx2 = -(theta[0]+theta[1]*maxx1)/theta[2]
    plt.plot([minx1, maxx1], [minx2, maxx2])
    plt.legend()
    plt.show()

#加载数据
data = np.loadtxt('ex2data1.txt', delimiter=',')

#提取数据
x, y = data[:, :-1], data[:, -1]
# x, y = np.split(data, [-1], axis=1)

#切割
m = x.shape[0]
num = int(m*0.7)
trainx, testx = np.split(x, [num])
trainy, testy = np.split(y, [num])

#预处理
def preprocess(x, y):
    #特征缩放
    meanx = np.mean(x, 0)
    sigma = np.std(x, 0, ddof=1)
    x = (x-meanx)/sigma
    m = x.shape[0]

    #预处理
    x = np.c_[np.ones((m, 1)), x]
    y = np.c_[y]
    return x, y

#进行预处理
trainx, trainy = preprocess(trainx, trainy)
testx, testy = preprocess(testx, testy)

#训练模型
jarr, theta = gradDesc(trainx, trainy)
testh = model(testx, theta)
trainh = model(trainx, theta)

#画图
draw(testx, testy, theta)
draw(trainx, trainy, theta)

#画代价曲线图
plt.title('代价曲线图')
plt.plot(jarr)
plt.show()

#sigmoid函数图
a = np.arange(-10, 10)
b = g(a)
plt.title('sigmoid函数图')
plt.plot(a, b)
plt.show()

print('测试集预测精度', accuracy(testh, testy))
print('训练集预测精度', accuracy(trainh, trainy))














