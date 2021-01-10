import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa


fig = plt.figure()
ax = fig.add_subplot(121)
plt.title('title aaa')
ax.matshow(samplemat((15, 15)), cmap='viridis')

ax1 = fig.add_subplot(122)
plt.title('title aaa')
ax1.matshow(samplemat((15, 15)), cmap='viridis')
plt.savefig('./mat_pdf.pdf')
