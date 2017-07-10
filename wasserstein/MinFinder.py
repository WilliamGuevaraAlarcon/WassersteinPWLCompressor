
import numpy as np
import matplotlib.pyplot as plt

def getfunctions(a,mu):
    def fun(x):
        f = a*(x-mu)*(x-mu)+np.sin(x)
        df = 2*a*(x-mu)+np.cos(x)
        print('\n fun( '+str(x)+' ) = '+str(f)+',  difffun( '+str(x)+' ) = '+str(df))
        return f, df
    return fun


def findmin(myfun, x1):
    f1, df1 = myfun(x1)
    if df1 == 0:
        return x1

    x2 = x1 - np.sign(df1)*(0.1 * x1)
    f2, df2 = myfun(x2)

    for i in range(7):
        print('x1: '+str(x1)+'   x2: '+str(x2)  )
        x3 = (df2*x1-df1*x2)/(df2-df1)
        f3, df3 = myfun(x3)

        if f2 < f1:
            x1, f1, df1 = x3, f3, df3
        else:
            x2, f2, df2 = x3, f3, df3




if __name__ == "__main__":
    myfun = getfunctions(2,4)
    findmin(myfun, 2.2)

    x = np.linspace(-2,9,100)
    f, df = myfun(x)

    plt.plot(x, f)

    plt.show()
    plt.plot(x, df)

    plt.show()
    #find bracket
