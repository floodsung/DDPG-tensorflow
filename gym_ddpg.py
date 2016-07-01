import gym
from ddpg import *
import gc
gc.enable()

ENV_NAME = 'Pendulum-v0'
EPISODES = 100000
STEPS = 200
TEST = 100

def main():
    env = gym.make(ENV_NAME)
    agent = DDPG(env)
    
    for episode in xrange(EPISODES):
        state = env.reset()
        print "episode:",episode
        # Train
        
        for step in xrange(STEPS):
            #env.render()
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 50 == 0 and episode > 100:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(STEPS):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
            if ave_reward >= -250:
                break
          
    # upload result
    env.monitor.start('gym_results/Pendulum-v0-experiment-1',force = True)
    for i in xrange(100):
    	total_reward = 0
        state = env.reset()
        for j in xrange(200):
            #env.render()
            action = agent.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            total_reward += reward
            if done:
                break
    ave_reward = total_reward/100
    print 'Evaluation Average Reward:',ave_reward
    env.monitor.close()

if __name__ == '__main__':
    main()
