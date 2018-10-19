import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("data/",one_hot=True)

COV_N=4;COV_STEP=2
w_cov=np.random.randn(COV_N,COV_N)
#b_cov=np.random.randn(COV_N**2)

P_N=(28-COV_N)/COV_STEP+(28-COV_N)%COV_STEP
p=np.zeros([P_N,P_N])
P_SIZE=2

FC_N=P_N/P_SIZE
fc=np.zeros([FC_N,FC_N])
w_fc=np.random.randn(FC_N**2,10)
b_fc=np.random.randn(10)
rate=0.01

for k in range(10000):
	x=mnist.train.images[k]
	x=np.reshape(x,[28,28])
    	for i in range(1,28,COV_STEP):
		for j in range(1,28,COV_STEP):
			x_b=np.zeros([COV_N,COV_N])
			x_b[0:COV_N,0:COV_N]=x[i:i+COV_N,j:j+COV_N]
			z=(x_b*w_cov).sum()
			p[i/COV_STEP][j/COV_STEP]=1/(1+np.exp(-z))
	for i in range(FC_N):
		for j in range(FC_N):
			fc[i][j]=p[i*P_SIZE:(i+1)*P_SIZE,j*P_SIZE:(j+1)*P_SIZE].max()
	z=np.matmul(fc.reshape(1,FC_N**2),w_fc)+b_fc
	o=1/(1+np.exp(-z))
	y=mnist.train.labels[k]
	dy=y-o
	do=(1-o)*o*dy
	dy_fc=np.matmul(do,w_fc.T)
	dy_p=np.zeros([P_N,P_N])   
	for i in range(FC_N):
		for j in range(FC_N):
			dy_p[i*P_SIZE:(i+1)*P_SIZE,j*P_SIZE:(j+1)*P_SIZE]=dy_fc.reshape([FC_N,FC_N])[i][j]/(P_SIZE**2)*np.ones([P_SIZE,P_SIZE])
	b_fc=b_fc+do
	   
	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[FC_N**2,1]),do)	  
	for i in range(1,28,COV_STEP):
		for j in range(1,28,COV_STEP):
			w_cov=w_cov+rate*x[i:i+COV_N,j:j+COV_N]*dy_p[i][j]*p[i][j](1-p[i][j])
			b_cov=b_cov+dy_p[i][j]


n=0

for k in range(100):
	x=mnist.test.images[k]
	x=np.reshape(x,[28,28])
        for i in range(1,28,COV_STEP):
                for j in range(1,28,COV_STEP):
                        x_b=x[i:i+COV_N,j:j+COV_N]
                        z=(x_b.reshape(1,COV_N**2)*w_cov.reshape(COV_N**2,1)).sum()
                        p[i][j]=1/(1+np.exp(-z))
        for i in range(FC_N):
                for j in range(FC_N):
                        fc[i][j]=p[i*P_SIZE:(i+1)*P_SIZE,j*P_SIZE:(j+1)*P_SIZE].max()
        z=np.matmul(fc.reshape(1,FC_N**2),w_fc)+b_fc
        o=1/(1+np.exp(-z))



	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



