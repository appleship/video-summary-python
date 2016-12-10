
import tensorflow as tf
import random
import numpy as np
import scipy.io as sio
'''
##################### the real y---frame store 0 or 1,the total length is 15%
matfn='/home/liu/test/summary/GT/Air_Force_One.mat'
data=sio.loadmat(matfn)
nframes=data['nFrames']
cframes=nframes*0.15
frame=np.zeros(nframes)
num=[random.randint(0, nframes) for i in range(cframes)]
print len(num)
for j in num:
	frame[j]=1 #the input y_label
'''
#############from the mat extract feature
matpath='/home/liu/test/summary/feature.mat'
original=sio.loadmat(matpath)
forword=[]
back=[]
for i in range(99):
	fea=original[str(i)][0]
	forword.append(fea)
	fea=original[str(98-i)][0]
	back.append(fea)

#############parameters

n_inputs =4096 
n_steps=99
n_hidden_units=3
batch_size=1

#########
x1 = tf.placeholder(tf.float32, shape=[None,4096])
x_input1=tf.reshape(x1,[-1,99,4096])
x2 = tf.placeholder(tf.float32, shape=[None,4096])
x_input2=tf.reshape(x2,[-1,99,4096])
#y = tf.placeholder(tf.float32, [None, 99])#the real one

############# first lstm
with tf.variable_scope('forward'):
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_input1, initial_state=init_state, time_major=False)
	outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))

##########second lstm
with tf.variable_scope('backward'):
	back_lstm = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	back_init = back_lstm.zero_state(batch_size, dtype=tf.float32)
	back_out, back_state = tf.nn.dynamic_rnn(back_lstm, x_input2, initial_state=back_init, time_major=False)
	back_out = tf.unpack(tf.transpose(back_out, [1, 0, 2]))

################ MLP layer

#input data  mlp_input[99 * 1 * 4102]
for i in range(n_steps):
	temp = tf.concat(1, [outputs[i], back_out[i]])	
	a=forword[i].reshape(1, 4096)#numpy
	put_in = tf.concat(1, [temp, tf.constant(a)])
	if i==0:
		mlp_input=put_in
	else:	
		mlp_input=tf.concat(0,[mlp_input,put_in])	
	
	

###########run
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
step=0
while step<1:	
	mlp_input=sess.run(mlp_input,feed_dict={x1:forword,x2:back})
 	print len(mlp_input)
	step=step+1
	
#print for_word[33] can extract the 33th tensor
#print sess.run(mlp_input), mlp_input.get_shape()

  
##############
