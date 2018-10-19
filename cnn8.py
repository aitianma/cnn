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
rate1=0.00000001
rate2=0.00000000000000001

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))


for k in range(10000):
	x=mnist.train.images[k]
	y=mnist.train.labels[k]

	x=np.reshape(x,[28,28])
	cov=signal.convolve2d(x,w_cov,"valid")
	cov=1/(1+np.exp(-cov))

	p=np.zeros([12,12])
        for i in range(12):
                for j in range(12):
                        p[i][j]=cov[i*2:(i+1)*2,j*2:(j+1)*2].max()

	cov1=signal.convolve2d(p,w_cov1,"valid")
        cov1=1/(1+np.exp(-cov1))


        p1=np.zeros([4,4])
        for i in range(4):
                for j in range(4):
                        p1[i][j]=cov1[i*2:(i+1)*2,j*2:(j+1)*2].max()
	fc=p1
	#fc=cov1

	z=np.matmul(fc.reshape(1,16),w_fc)+b_fc	
	o=1/(1+np.exp(-z))
	dy=y-o
	do=o*(1-o)*dy
	w_fc=w_fc+rate*np.matmul(np.reshape(fc,[16,1]),np.reshape(do,[1,10]))
	b_fc=b_fc+do

	dy_fc=np.matmul(np.reshape(do,[1,10]),w_fc.T)
	dy_p1=dy_fc.reshape([4,4])
	dy_cov1=np.zeros([8,8])   
	for i in range(4):
		for j in range(4):
			dy_cov1[i*2:(i+1)*2,j*2:(j+1)*2]=dy_p1[i][j]*cov1[i*2:(i+1)*2,j*2:(j+1)*2]
	#w_cov1=w_cov1+rate1*signal.convolve2d(p[::-1],dy_cov1*cov1*(1-cov1),"valid")

	dy_p=signal.convolve2d(dy_cov1,FZ(w_cov1),"full")
	#dy_p=dy_p[::-1]
	p_rot_180=FZ(p)
#	for i in range(8):
#		for j in range(8):	
#			w_cov1=w_cov1+rate1*p_rot_180[i:i+5,j:j+5]*dy_cov1[i][j]

	w_cov1=w_cov1+rate1*signal.convolve2d(dy_cov1,FZ(p),"valid")


	dy_cov=np.zeros([24,24])
	for i in range(12):
		for j in range(12):
			dy_cov[i*2:(i+1)*2,j*2:(j+1)*2]=cov[i*2:(i+1)*2,j*2:(j+1)*2]*dy_p[i][j]


	w_cov=w_cov+rate2*signal.convolve2d(dy_cov,FZ(x),"valid")
	

	#x_rot_180=FZ(x) 
	#for i in range(24):
	#	for j in range(24):
	#		w_cov=w_cov+rate2*x_rot_180[i:i+5,j:j+5]*dy_cov[i][j]
		#	b_cov=b_cov+dy_p[i][j]*p[i][j]*(1-p[i][j])


	#print w_cov
	#print w_fc
n=0

w_cov.dump("/tmp/w_cov")
w_fc.dump("/tmp/w_fc")

for k in range(1000):
	x=mnist.test.images[k]

	x=np.reshape(x,[28,28])
        cov=signal.convolve2d(x,w_cov,"valid")
        cov=1/(1+np.exp(-cov))

        p=np.zeros([12,12])
        for i in range(12):
                for j in range(12):
                        p[i][j]=cov[i*2:(i+1)*2,j*2:(j+1)*2].max()

        cov1=signal.convolve2d(p,w_cov1,"valid")
        cov1=1/(1+np.exp(-cov1))

        p1=np.zeros([4,4])
        for i in range(4):
                for j in range(4):
                        p1[i][j]=cov1[i*2:(i+1)*2,j*2:(j+1)*2].max()
        fc=p1

        z=np.matmul(fc.reshape(1,16),w_fc)+b_fc
        o=1/(1+np.exp(-z))

	if np.argmax(o)==np.argmax(mnist.test.labels[k]):
		n=n+1


print n

	



