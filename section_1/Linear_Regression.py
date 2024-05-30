import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)


def f(x):
    return 3*x + 5

# create data
x = np.linspace(0,10,50).reshape(50,1) # numpy array
y = f(x).reshape(50,1)                # numpy array



# add noice
def f_noice(x):
    noice = np.random.randn(50,1)
    return 3*(x+noice) + 5

y_noice = f_noice(x).reshape(50,1)
bar = np.ones((50,1))
xbar = np.concatenate((bar, x), axis=1)


# closed-form solution
a = xbar.T.dot(xbar)
a = np.linalg.pinv(a)
b = xbar.T.dot(y_noice)
w_closed = np.dot(a,b)

w_0_clo = w_closed[0][0] # constant
w_1_clo = w_closed[1][0] # coefficiens



# extend solution
def lossMSE(y, xbar, w):
    return 1/2*np.mean((y - xbar.dot(w))**2)

def gradMSE(xbar, y, w):
    a = xbar.dot(w)
    a = y-a
    return -xbar.T.dot(a)


def fitMSE(xbar,y,w,eta=0.001):
    w_save = [w]

    for i in range(10000):
        w_new = w_save[-1] - eta*gradMSE(xbar,y,w_save[-1])
        w_save.append(w_new)
        if (i%1000 == 0):
            loss = lossMSE(y,xbar,w_save[-1])
            print("loss : ",loss)

    return w_save





w_extend = np.random.randn(2,1).reshape(2,1)

w_save = fitMSE(xbar, y,w_extend)


w_0_ex = w_save[-1][0][0]  # constant
w_1_ex = w_save[-1][1][0]  # coefficiens


print("closed-form: ", w_closed)
print("extend-form: ", w_save[-1])

plt.subplot(1,2,1)
plt.plot(x,y_noice, "o")
plt.plot(x,w_1_clo*x+w_0_clo, "r")
plt.title("using closed-form solution")
plt.text(5, 5, 'w : {}'.format(w_closed), fontsize = 10)

plt.subplot(1,2,2)
plt.plot(x,y_noice, "o")
plt.plot(x,w_1_ex*x + w_0_ex, "r")
plt.title("using gradient decent")
plt.text(5, 5, 'w : {}'.format(w_save[-1]), fontsize = 10)

plt.savefig('linear_regression.png', bbox_inches='tight')
plt.show()