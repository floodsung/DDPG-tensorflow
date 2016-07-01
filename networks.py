import tensorflow as tf
import numpy as np

def fanin_init(shape,fanin=None):
  fanin = fanin or shape[0]
  v = 1/np.sqrt(fanin)
  return tf.random_uniform(shape,minval=-v,maxval=v)


l1 = 400 # dm 400
l2 = 300 # dm 300


def theta_p(state_dim,action_dim):
  with tf.variable_scope("theta_p"):
    return [tf.Variable(fanin_init([state_dim,l1]),name='1w'),
            tf.Variable(fanin_init([l1],state_dim),name='1b'),
            tf.Variable(fanin_init([l1,l2]),name='2w'),
            tf.Variable(fanin_init([l2],l1),name='2b'),
            tf.Variable(tf.random_uniform([l2,action_dim],-3e-3,3e-3),name='3w'),
            tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3),name='3b')]
  
def policy_network(state,theta,name='policy'):
  with tf.variable_op_scope([state],name,name):
    h0 = tf.identity(state,name='h0-state')
    h1 = tf.nn.relu( tf.matmul(h0,theta[0]) + theta[1],name='h1')
    h2 = tf.nn.relu( tf.matmul(h1,theta[2]) + theta[3],name='h2')
    h3 = tf.identity(tf.matmul(h2,theta[4]) + theta[5],name='h3')
    action = tf.nn.tanh(h3,name='h4-action')
    return action


def theta_q(state_dim,action_dim):
  with tf.variable_scope("theta_q"):
    return [tf.Variable(fanin_init([state_dim,l1]),name='1w'),
            tf.Variable(fanin_init([l1],state_dim),name='1b'),
            tf.Variable(fanin_init([l1+action_dim,l2]),name='2w'),
            tf.Variable(fanin_init([l2],l1+action_dim),name='2b'),
            tf.Variable(tf.random_uniform([l2,1],-3e-3,3e-3),name='3w'),
            tf.Variable(tf.random_uniform([1],-3e-3,3e-3),name='3b')]
    
def q_network(state,action,theta, name="q_network"):
  with tf.variable_op_scope([state,action],name,name):
    h0 = tf.identity(state,name='h0-state')
    h0a = tf.identity(action,name='h0-act')
    h1  = tf.nn.relu( tf.matmul(h0,theta[0]) + theta[1],name='h1')
    h1a = tf.concat(1,[h1,action])
    h2  = tf.nn.relu( tf.matmul(h1a,theta[2]) + theta[3],name='h2')
    qs  = tf.matmul(h2,theta[4]) + theta[5]
    q = tf.squeeze(qs,[1],name='h3-q')
    
    return q