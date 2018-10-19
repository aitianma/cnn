import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("data/",one_hot=True)

w_cov=np.random.randn(4,4)
w_fc=np.random.randn(25,10)
b_cov=np.random.randn(16)
b_fc=np.random.randn(10)
rate=0.9
for k in range(5000):
	x=mnist.train.images[k]
	x=np.reshape(x,[28,28])
	p=np.zeros([25,25])
    	for i in range(25):
		for j in range(25):
			x_b=x[i:i+4,j:j+4]
			z=(x_b.reshape(1,16)*w_cov.reshape(16,1)+b_cov).sum()/16
			#p[i][j]=1/(z+np.exp(-z))
	fc=np.zeros([5,5])
	for i in range(5):
		for j in range(5):
			fc[i][j]=p[i:i+4,j:j+4].max()
	z=np.matmul(fc.reshape(1,25),w_fc)+b_fc
	o=1/(1+np.exp(-z))
	y=mnist.train.labels[k]
	dy=y-o
	do=o*(1-o)
	#dy_fc=np.matmul((1-o)*o*dy,w_fc.T)
	#dy_p=np.zeros([25,25])   
	#for i in range(5):
	#	for j in range(5):
	#		dy_p[i:i+5,j:j+5]=np.ones([5,5])*dy_fc.reshape([5,5])[i][j]/25
	b_fc=b_fc+dy*do
	   
	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[25,1]),dy*do)	  
	#for i in range(25):
	#	for j in range(25):
	#		w_cov=w_cov+rate*x[i:i+4,j:j+4]*dy_p[i][j]
	#		b_cov=b_cov+dy_p[i][j]/16


n=0

for k in range(1000):
	x=mnist.test.images[k]
	x=np.reshape(x,[28,28])
        p=np.zeros([25,25])
        for i in range(25):
                for j in range(25):
                        x_b=x[i:i+4,j:j+4]
			z=(x_b.reshape(1,16)*w_cov.reshape(16,1)+b_cov).sum()/16
                        #p[i][j]=1/(z+np.exp(-z))
                        
        fc=np.zeros([5,5])
        for i in range(5):
                for j in range(5):
                        fc[i][j]=p[i:i+4,j:j+4].max()
        z=np.matmul(fc.reshape(1,25),w_fc)+b_fc
        o=1/(1+np.exp(-z))

	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



