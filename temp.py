import os
import struct
import numpy as np
from scipy import misc


num = misc.imread('/home/aditaker/Downloads/English/Fnt/Sample010/img010-00017.png')
k = misc.imresize(num, (28, 28))
q = np.full((28, 28), 255)
k=q-k
print np.shape(k)
X=np.reshape(k,(1,784))
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

syn0 = np.loadtxt('syn0.out', usecols=range(100))
syn1 = np.loadtxt('syn1.out', usecols=range(50))
syn2 = np.loadtxt('syn2.out', usecols=range(10))
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return np.nan_to_num( 1/(1+np.exp(-x)) )

l0 = X
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))
l3 = nonlin(np.dot(l2,syn2))
# how much did we miss the target value?
show(k)
print l3
a=np.argmax(l3)

print a

