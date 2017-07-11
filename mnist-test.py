import os
import struct
import numpy as np

lbl=[]
img=[]
def read(dataset = "training", path = "."):

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    global lbl
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    global img
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)



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

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
    
read(dataset="testing")

X=np.reshape(img,(10000,784))
print np.shape(X)
# randomly initialize our weights with mean 0
#syn0 = 2*np.random.random((784,100)) - 1
#syn1 = 2*np.random.random((100,10)) - 1
syn0 = np.loadtxt('syn0.out', usecols=range(100))
syn1 = np.loadtxt('syn1.out', usecols=range(100))
syn2 = np.loadtxt('syn2.out', usecols=range(10))
print np.shape(syn0)
print np.shape(syn1)
count=0
for j in xrange(10000):


    l0 = X[j]
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
    # how much did we miss the target value?
   
    k=np.argmax(l3)
    
    if (k-lbl[j])==0:
        count=count+1
print count
