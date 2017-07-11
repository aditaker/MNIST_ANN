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
        gradients = 1. * (x > 0)
        gradients[gradients == 0] = 0.01
        return gradients

    return np.maximum(0.01*x,x)
    
read()

mean=0
X=np.reshape(img,(60000,784))
for i in xrange(60000):
    mean=(mean+X[i])/60000;
for i in xrange(60000):
    X[i]-=mean  
y=np.zeros((60000,10))
for i in xrange(60000):
    y[i][lbl[i]]=1
print X.shape
print y.shape

np.random.seed(1)

# randomly initialize our weights with mean 0
#syn0 = 2*np.random.random((784,100)) - 1
syn0 = np.random.normal(0,1/np.sqrt(100),(784, 100))
#syn1 = 2*np.random.random((100,10)) - 1
syn1 = np.random.normal(0,1/np.sqrt(100),(100, 100))
syn2 = np.random.normal(0,1/np.sqrt(10),(100, 10))
rate=0.001
for j in xrange(25):

    # Feed forward through layers 0, 1, and 2
    batches = [ X[x:x+10] \
                        for x in np.arange(0, 60000, 10) ]
    outs = [ y[x:x+10] \
                        for x in np.arange(0, 60000, 10) ]
    p=0
    for batch in batches:
        l0 = batch
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        l3 = nonlin(np.dot(l2,syn2))

        l3_error = l3-outs[p]
        p=p+1;
          
        l3_delta = l3_error*nonlin(l3,deriv=True)

        l2_error = l3_delta.dot(syn2.T)
        
        l2_delta = l2_error * nonlin(l2,deriv=True)

        l1_error = l2_delta.dot(syn1.T)
        
        l1_delta = l1_error * nonlin(l1,deriv=True)

        syn2 -= rate*(l2.T.dot(l3_delta))
        syn1 -= rate*(l1.T.dot(l2_delta))
        syn0 -= rate*(l0.T.dot(l1_delta))

    print "Epoch"
    count=0
    for j in xrange(60000):
        l0 = X[j]
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        l3 = nonlin(np.dot(l2,syn2))
        # how much did we miss the target value?
       
        k=np.argmax(l3)
        
        if (k-lbl[j])==0:
            count=count+1

    print count

    



np.savetxt('syn0.out', syn0)
np.savetxt('syn1.out', syn1)
np.savetxt('syn2.out', syn2)