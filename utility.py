
import tensorflow as tf 
import math


def input(shape):
	return tf.placeholder("float",shape)

# f fan-in size
def variable(shape,f):
	return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


