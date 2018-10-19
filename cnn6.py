import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from scipy import signal
mnist=input_data.read_data_sets("data/",one_hot=True)

w_cov=np.random.randn(5,5)
w_cov1=np.random.randn(5,5)
w_fc=np.random.randn(16,10)
#b_cov=np.zeros(25)
b_fc=np.zeros(10)
rate=0.8

for k in range(10000):
	x=mnist.train.images[k]
	y=mnist.train.labels[k]

	x=np.reshape(x,[28,28])
	p=signal.convolve2d(x,w_cov,"valid")
	p=1/(1+np.exp(-p))

	fc=np.zeros([12,12])
        for i in range(12):
                for j in range(12):
                        fc[i][j]=p[i*2:(i+1)*2,j*2:(j+1)*2].max()

	p=signal.convolve2d(fc,w_cov1,"valid")
        p=1/(1+np.exp(-p))

        fc=np.zeros([4,4])
        for i in range(4):
                for j in range(4):
                        fc[i][j]=p[i*2:(i+1)*2,j*2:(j+1)*2].max()

	z=np.matmul(fc.reshape(1,16),w_fc)+b_fc	
	o=1/(1+np.exp(-z))
	dy=y-o
	do=o*(1-o)*dy
	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[16,1]),np.reshape(do,[1,10]))
	b_fc=b_fc+do

#	dy_fc=np.matmul(np.reshape(do*dy,[1,10]),w_fc.T)
#	dy_p=np.zeros([24,24])   
#	for i in range(12):
#		for j in range(12):
#			dy_p[i*2:(i+1)*2,j*2:(j+1)*2]=dy_fc.reshape([4,4])[i][j]/4*np.ones([2,2])
	   
#	for i in range(24):
#		for j in range(24):
#			w_cov=w_cov+rate*x[i:i+5,j:j+5]*dy_p[i][j]*p[i][j]*(1-p[i][j])
		#	b_cov=b_cov+dy_p[i][j]*p[i][j]*(1-p[i][j])


	#print w_cov
	#print w_fc
n=0

w_cov.dump("/tmp/w_cov")
w_fc.dump("/tmp/w_fc")

for k in range(1000):
	x=mnist.test.images[k]

	x=np.reshape(x,[28,28])
        p=signal.convolve2d(x,w_cov,"valid")
        p=1/(1+np.exp(-p))

        fc=np.zeros([12,12])
        for i in range(12):
                for j in range(12):
                        fc[i][j]=p[i*2:(i+1)*2,j*2:(j+1)*2].max()

        p=signal.convolve2d(fc,w_cov1,"valid")
        p=1/(1+np.exp(-p))

        fc=np.zeros([4,4])
        for i in range(4):
                for j in range(4):
                        fc[i][j]=p[i*2:(i+1)*2,j*2:(j+1)*2].max()

        z=np.matmul(fc.reshape(1,16),w_fc)+b_fc
        o=1/(1+np.exp(-z))

	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



