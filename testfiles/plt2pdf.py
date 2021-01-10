import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['font.family'] = 'Calibri'
#
# if __name__ == '__main__' :
#     t1 = np.arange(0, 5, 0.1)
#     t2 = np.arange(0, 5, 0.02)
#
#     plt.figure(12)
#     plt.subplot(221)
#     plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
#
#     plt.subplot(222)
#     plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
#
#     plt.subplot(212)
#     plt.title("212")
#     plt.plot( [1, 2, 3, 4], [1, 4, 9, 16])
#
#     plt.savefig('./test_pdf.pdf')
#     plt.show()

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号
mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

s = ['221', '222', '122']

#需要画的变量的名字都放在factorList里面，写一个画图（折线图）的函数plot_factor，需要三个参数
#但是这个画图函数里面没有写plt.close(),而是在循环时候写在函数外。

with PdfPages('test_pdf.pdf') as pdf:
    for i in range(3):
        # plot_factor(factorList[i], xaxislabel[i], dateSeries)
        t1 = np.arange(0, 5, 0.1)
        t2 = np.arange(0, 5, 0.02)

        plt.figure(12)
        plt.subplot(221)
        plt.text(1, 1, s[0])
        plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')

        plt.subplot(222)
        plt.text(2, 2, s[1])
        plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

        plt.subplot(122)
        plt.title("212")
        plt.text(1, 1, s[2])
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        pdf.savefig()
        plt.close()

