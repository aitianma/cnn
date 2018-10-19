import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from scipy import signal
mnist=input_data.read_data_sets("data/",one_hot=True)

w_cov=np.random.randn(2,3,3)
w_cov1=np.random.randn(3,3)
w_fc=np.random.randn(2*24*24,10)


b_fc=np.random.randn(10)
rate=0.01
rate1=0.0000001
rate2=0.0000000000001

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))


def sigmoid(a):
	return 1/(1+np.exp(-a))

def sigmoid_der(a):
	return a*(1-a)

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
	cov=signal.convolve2d(x,w_cov[0],"valid")
	cov=sigmoid(cov)

	
	cov_1=signal.convolve2d(x,w_cov[1],"valid")
        cov_1=sigmoid(cov_1)


	cov1=signal.convolve2d(cov,w_cov1,"valid")
        cov1=sigmoid(cov1)

	cov1_1=signal.convolve2d(cov_1,w_cov1,"valid")
        cov1_1=sigmoid(cov1_1)

	fc=np.append(cov1,cov1_1)

	z=np.matmul(fc.reshape(1,2*24*24),w_fc)+b_fc	
	o=sigmoid(z)

	dy=y-o
	dy_o=dy*sigmoid_der(o)

	loss[k]=np.sqrt((dy*dy).sum())

	dy_fc=np.matmul(np.reshape(dy_o,[1,10])[0],w_fc.T)
	dy_fc1=dy_fc[0:24*24] 
	dy_fc2=dy_fc[24*24:2*24*24]
	dy_fc1=np.reshape(dy_fc1,[24,24])
	dy_fc2=np.reshape(dy_fc2,[24,24])
	dy_cov1=signal.convolve2d(dy_fc1,FZ(w_cov1),"full")
	dy_cov1_1=signal.convolve2d(dy_fc2,FZ(w_cov1),"full")


	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[24*24*2,1]),np.reshape(dy_o,[1,10]))
	b_fc=b_fc+dy_o

	w_cov1=w_cov1+rate1*FZ(signal.convolve2d(cov,FZ(dy_fc1*sigmoid_der(cov1)),"valid"))
	w_cov1=w_cov1+rate1*FZ(signal.convolve2d(cov_1,FZ(dy_fc2*sigmoid_der(cov1_1)),"valid"))	

	w_cov[0]=w_cov[0]+rate2*FZ(signal.convolve2d(x,FZ(dy_cov1*sigmoid_der(cov)),"valid"))
	w_cov[1]=w_cov[1]+rate2*FZ(signal.convolve2d(x,FZ(dy_cov1*sigmoid_der(cov_1)),"valid"))	


n=0

w_cov.dump("/tmp/w_cov")
w_cov1.dump("/tmp/w_cov1")
w_fc.dump("/tmp/w_fc")
loss.dump("/tmp/loss")

for k in range(1000):
	x=mnist.test.images[k]

	x=x-x.mean()
        x=np.reshape(x,[28,28])
        cov=signal.convolve2d(x,w_cov[0],"valid")
        cov=sigmoid(cov)


        cov_1=signal.convolve2d(x,w_cov[1],"valid")
        cov_1=sigmoid(cov_1)


        cov1=signal.convolve2d(cov,w_cov1,"valid")
        cov1=sigmoid(cov1)

        cov1_1=signal.convolve2d(cov_1,w_cov1,"valid")
        cov1_1=sigmoid(cov1_1)

        fc=np.append(cov1,cov1_1)

        z=np.matmul(fc.reshape(1,2*24*24),w_fc)+b_fc
        o=sigmoid(z)


	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



