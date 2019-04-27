import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#sigmoid函数
def g(z, deriv=False):
    if deriv == True:
        return z*(1-z)
    return 1.0/(1+np.exp(-z))

#模型
def model(x ,theta1, theta2):
    z2 = x.dot(theta1)
    a2 = g(z2)
    z3 = a2.dot(theta2)
    a3 = g(z3)
    return a2, a3

#代价函数
def costfunc(h, y):
    m = len(h)
    J = -1.0/m*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
    return J

#精度
def score(h ,y):
    #多分类，转换为多个二分类问题，[0.1, 0.7, 0.1, 0.1] = [0, 1, 0, 0]
    m = len(h)
    count = 0
    for i in range(m):
        if np.argmax(h[i]) == np.argmax(y[i]):
            count += 1
    return count/m

#BP算法
def BP(a1, a2, a3, y, theta1, theta2, alpha):
    delta3 = a3-y #反向转播，误差
    delta2 = delta3.dot(theta2.T)*g(a2, deriv=True)
    m = len(a1)
    deltatheta1 = 1.0/m*a1.T.dot(delta2)
    deltatheta2 = 1.0/m*a2.T.dot(delta3)

    theta1 -= deltatheta1*alpha
    theta2 -= deltatheta2*alpha
    return theta1, theta2

#梯度下降最优解
def gradDesc(x, y, hidden_layer_sizes=(17, ), max_iter=1500, alpha=0.1):
    m, n = x.shape #x与theta相乘
    col = y.shape[1]#标签数
    theta1 = 2*np.random.rand(n, hidden_layer_sizes[0])-1
    theta2 = 2*np.random.rand(hidden_layer_sizes[0], col)-1 #行与theta1结果对应，列于标签个数对应
    jarr = np.zeros(max_iter)

    for i in  range(max_iter):
        a2, a3 = model(x, theta1, theta2)
        jarr[i] = costfunc(a3, y)
        theta1, theta2 = BP(x, a2, a3, y, theta1, theta2, alpha)
    return jarr, theta1, theta2

#预处理
def preprocess(x, y):
    #标准缩放
    meanx = np.mean(x)
    sigma = np.std(x, ddof=1)
    x = (x-meanx)/sigma
    m = len(x)
    #预处理
    x = np.c_[np.ones((m, 1)), x]
    y = np.c_[y]
    return x, y

#加载数据
x = np.loadtxt('imgX.txt', delimiter=',')
y = np.loadtxt('labely.txt', delimiter=',')
y[y == 10] = 0
x, y = preprocess(x, y)

#多分类转为多个二分类问题
coder = OneHotEncoder(categories='auto') #categories类别 auto自动
y = coder.fit_transform(y).toarray()  #coder编码器 fit配合 transform 改变
#将y变为0和1 

#洗牌，洗牌整体下标对应元素改变，不影响预处理
m = len(x)
np.random.seed(0)
order = np.random.permutation(m) #permutation置换
x = x[order]
y = y[order]
n = x.shape[0]

#切分
num = int(m*0.7)
trainx, testx = np.split(x, [num])
trainy, testy = np.split(y, [num])

#训练模型
jarr, theta1, theta2 = gradDesc(trainx, trainy, hidden_layer_sizes=(80, ), max_iter=2000)

#计算预测值
a2, trainh = model(trainx, theta1, theta2)
a2, testh = model(testx, theta1, theta2)

plt.plot(jarr)
plt.show()

print('train score', score(trainy, trainh))
print('test score', score(testy, testh))

#将OneHotEncoder(编码)转为数值型,以正常画出混淆矩阵
testy = np.argmax(testy, axis=1)
testh = np.argmax(testh, axis=1)

print('混淆矩阵:\n', confusion_matrix(testy, testh))
print('分类报告:\n', classification_report(testy, testh))













