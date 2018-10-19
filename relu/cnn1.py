import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from scipy import signal
mnist=input_data.read_data_sets("data/",one_hot=True)

w_cov=np.random.randn(3,3)
w_fc=np.random.randn(26*26,10)


b_fc=np.random.randn(10)
rate=0.5
rate1=0.000000000001

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))


def sigmoid(a):
	return 1/(1+np.exp(-a))

def sigmoid_der(a):
	return sigmoid(a)*(1-sigmoid(a))

def relu(a):
	return np.maximum(a,0)

def relu_der(a):
	a[a>0]=1
	a[a<=0]=0
	return a
loss=np.random.randn(5000)

for k in range(5000):

	x=mnist.train.images[k]
	y=mnist.train.labels[k]

	x=x-x.mean()
	x=np.reshape(x,[28,28])
	cov_z=signal.convolve2d(x,FZ(w_cov),"valid")
	cov=relu(cov_z)
	
	fc=cov

	z=np.matmul(fc.reshape(1,26*26),w_fc)+b_fc	
	o=sigmoid(z)

	dy=y-o
	dy_o=dy*sigmoid_der(z)

	loss[k]=np.sqrt((dy*dy).sum())

	dy_fc=np.matmul(np.reshape(dy_o,[1,10])[0],w_fc.T)
	dy_fc=np.reshape(dy_fc,[26,26])
	dy_cov=signal.convolve2d(dy_fc,w_cov,"full")


	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[26*26,1]),np.reshape(dy_o,[1,10]))
	b_fc=b_fc+dy_o

	w_cov=w_cov+rate1*signal.convolve2d(x,FZ(dy_fc*relu_der(cov_z)),"valid")


n=0

w_cov.dump("/tmp/w_cov")
w_fc.dump("/tmp/w_fc")
loss.dump("/tmp/loss")

for k in range(1000):
	x=mnist.test.images[k]

	x=x-x.mean()
        x=np.reshape(x,[28,28])
        cov_z=signal.convolve2d(x,FZ(w_cov),"valid")
        cov=relu(cov_z)

        fc=cov

        z=np.matmul(fc.reshape(1,26*26),w_fc)+b_fc
        o=sigmoid(z)


	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



