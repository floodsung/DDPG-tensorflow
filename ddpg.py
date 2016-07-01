# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from ou_noise import OUNoise
import networks

# Hyper Parameters:
REPLAY_BUFFER_SIZE = 500000
REPLAY_START_SIZE = 20000
BATCH_SIZE = 32
GAMMA = 0.99
LAYER1_SIZE = 400
LAYER2_SIZE = 300
Q_LEARNING_RATE = 0.001
P_LEARNING_RATE = 0.0001
TAU = 0.001
L2_Q = 0.01
L2_POLICY = 0.0

class DDPG:
	def __init__(self, env):
		self.name = 'DDPG' # name for uploading results
		self.environment = env
		
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]
		# Initialize time step
		self.time_step = 0
		# initialize replay buffer
		self.replay_buffer = deque()
		# initialize networks
		self.create_networks_and_training_method(state_dim,action_dim)

		self.sess = tf.InteractiveSession()
		self.sess.run(tf.initialize_all_variables())

		# loading networks
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
				print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
				print "Could not find old network weights"

		global summary_writer
		summary_writer = tf.train.SummaryWriter('~/logs',graph=self.sess.graph)
	
	def create_networks_and_training_method(self,state_dim,action_dim):

		theta_p = networks.theta_p(state_dim,action_dim)
		theta_q = networks.theta_q(state_dim,action_dim)
		target_theta_p,target_update_p = self.exponential_moving_averages(theta_p,TAU)
		target_theta_q,target_update_q = self.exponential_moving_averages(theta_q,TAU)

		self.state = tf.placeholder(tf.float32,[None,state_dim],'state')
		self.action_test = networks.policy_network(self.state,theta_p)

		# Initialize a random process the Ornstein-Uhlenbeck process for action exploration
		self.exploration = OUNoise(action_dim)
		noise = self.exploration.noise()
		self.action_exploration = self.action_test + noise

		q = networks.q_network(self.state,self.action_test,theta_q)
		# policy optimization
		mean_q = tf.reduce_mean(q)
		weight_decay_p = tf.add_n([L2_POLICY * tf.nn.l2_loss(var) for var in theta_p])  
		loss_p = -mean_q + weight_decay_p

		optim_p = tf.train.AdamOptimizer(P_LEARNING_RATE)
		grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=theta_p)
		optimize_p = optim_p.apply_gradients(grads_and_vars_p)
		with tf.control_dependencies([optimize_p]):
			self.train_p = tf.group(target_update_p)

		# q optimization
		self.action_train = tf.placeholder(tf.float32,[None,action_dim],'action_train')
		self.reward = tf.placeholder(tf.float32,[None],'reward')
		self.next_state = tf.placeholder(tf.float32,[None,state_dim],'next_state')
		self.done = tf.placeholder(tf.bool,[None],'done')

		q_train = networks.q_network(self.state,self.action_train,theta_q)
		next_action = networks.policy_network(self.next_state,theta=target_theta_p)
		next_q = networks.q_network(self.next_state,next_action,theta=target_theta_q)
		q_target = tf.stop_gradient(tf.select(self.done,self.reward,self.reward + GAMMA * next_q))

		# q loss
		q_error = tf.reduce_mean(tf.square(q_target - q_train))
		weight_decay_q = tf.add_n([L2_Q * tf.nn.l2_loss(var) for var in theta_q])
		loss_q = q_error + weight_decay_q

		optim_q = tf.train.AdamOptimizer(Q_LEARNING_RATE)
		grads_and_vars_q = optim_q.compute_gradients(loss_q, var_list=theta_q)
		optimize_q = optim_q.apply_gradients(grads_and_vars_q)
		with tf.control_dependencies([optimize_q]):
			self.train_q = tf.group(target_update_q)

		tf.scalar_summary("loss_q",loss_q)
		tf.scalar_summary("loss_p",loss_p)
		tf.scalar_summary("q_mean",mean_q)
		global merged_summary_op
		merged_summary_op = tf.merge_all_summaries()

	def train(self):
		#print "train step",self.time_step
		# Sample a random minibatch of N transitions from replay buffer
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]
		done_batch = [data[4] for data in minibatch]

		_,_,summary_str = self.sess.run([self.train_p,self.train_q,merged_summary_op],feed_dict={
			self.state:state_batch,
			self.action_train:action_batch,
			self.reward:reward_batch,
			self.next_state:next_state_batch,
			self.done:done_batch
			})

		summary_writer.add_summary(summary_str,self.time_step)

		# save network every 1000 iteration
		if self.time_step % 1000 == 0:
			self.saver.save(self.sess, 'saved_networks/' + 'network' + '-ddpg', global_step = self.time_step)

	def noise_action(self,state):
		# Select action a_t according to the current policy and exploration noise
		action = self.sess.run(self.action_exploration,feed_dict={
			self.state:[state]
			})[0]
		return np.clip(action,self.environment.action_space.low,self.environment.action_space.high)

	def action(self,state):
		action = self.sess.run(self.action_test,feed_dict={
			self.state:[state]
			})[0]
		return np.clip(action,self.environment.action_space.low,self.environment.action_space.high)

	def perceive(self,state,action,reward,next_state,done):
		# Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
		self.replay_buffer.append((state,action,reward,next_state,done))
		# Update time step
		self.time_step += 1

		# Limit the replay buffer size
		if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
			self.replay_buffer.popleft()

		# Store transitions to replay start size then start training
		if self.time_step >  REPLAY_START_SIZE:
			self.train()

		# Re-iniitialize the random process when an episode ends
		if done:
			self.exploration.reset()

	# f fan-in size
	def exponential_moving_averages(self,theta, tau=0.001):
		ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
		update = ema.apply(theta)  # also creates shadow vars
		averages = [ema.average(x) for x in theta]
		return averages, update