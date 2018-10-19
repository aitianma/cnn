import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("data/",one_hot=True)

w_cov=np.random.randn(9,9)
w_fc=np.random.randn(100,10)
#w_cov=np.load("/tmp/w_cov")
#w_fc=np.load("/tmp/w_fc")
#b_cov=np.zeros(25)
b_fc=np.zeros(10)
rate=0.1

for k in range(2000):
	x=mnist.train.images[k]
	y=mnist.train.labels[k]

	x=np.reshape(x,[28,28])
	p=np.zeros([20,20])
	for i in range(20):
		for j in range(20):
			x_b=x[i:i+9,j:j+9]
			z=(x_b*w_cov).sum()
			p[i][j]=1/(1+np.exp(-z))
	fc=np.zeros([10,10])
	for i in range(10):
		for j in range(10):
			fc[i][j]=p[i*2:(i+1)*2,j*2:(j+1)*2].max()
	z=np.matmul(fc.reshape(1,100),w_fc)+b_fc	
	o=1/(1+np.exp(-z))
	dy=y-o
	do=o*(1-o)
	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[100,1]),np.reshape(dy*do,[1,10]))
	b_fc=b_fc+dy*do

	dy_fc=np.matmul(np.reshape(do*dy,[1,10]),w_fc.T)
	dy_p=np.zeros([20,20])   
	for i in range(10):
		for j in range(10):
			dy_p[i*2:(i+1)*2,j*2:(j+1)*2]=dy_fc.reshape([10,10])[i][j]/4*p[i*2:(i+1)*2,j*2:(j+1)*2]
	   
	for i in range(20):
		for j in range(20):
			w_cov=w_cov+rate*x[i:i+9,j:j+9]*dy_p[i][j]*p[i][j]*(1-p[i][j])
		#	b_cov=b_cov+dy_p[i][j]*p[i][j]*(1-p[i][j])


	#print w_cov
	#print w_fc
n=0

w_cov.dump("/tmp/w_cov")
w_fc.dump("/tmp/w_fc")

for k in range(100):
	x=mnist.test.images[k]
	x=np.reshape(x,[28,28])
        p=np.zeros([20,20])
        for i in range(20):
                for j in range(20):
                        x_b=x[i:i+9,j:j+9]
                        z=(x_b*w_cov).sum()
                        p[i][j]=1/(1+np.exp(-z))
        fc=np.zeros([10,10])
        for i in range(10):
                for j in range(10):
                        fc[i][j]=p[i*2:(i+1)*2,j*2:(j+1)*2].max()
        z=np.matmul(fc.reshape(1,100),w_fc)+b_fc
        o=1/(1+np.exp(-z))


	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



