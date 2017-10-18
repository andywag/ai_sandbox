
""" This module contains a basic test for controlling the OpenAI cartpoll ENV. It is very simple and used only for
    an initial experiment in controllign the environment
"""


import gym
import matplotlib.pyplot as plot
import numpy as np


env = gym.make('CartPole-v0')

def getAction(model,observation) :
    ### Returns the action of the linear mode - sgn(model*observation)"""
    action = 0 if np.matmul(model, observation) < 0 else 1
    return action

def guess():
    ### Simple guessing of the linear model for the system ###
    observation = env.reset()
    model = np.random.randn(4)
    for x in range(200):
        #env.render()
        act = getAction(model,observation)
        observation, reward, done, info = env.step(act)
        if (done) :
            return x+1
    return 200

def descend():
    ### Basic gradient descent algorithm which updates the model depending on the quality of result ###
    lastReward = 0
    bestModel  = np.random.randn(4)
    alpha = .15
    observation = env.reset()
    delta = np.random.randn(4)

    for x in range(200):

        model = bestModel + alpha*delta
        act = getAction(model,observation)
        observation, reward, done, info = env.step(act)
        if (x > lastReward) :
            lastReward = x
            bestModel = model

        if (done) :
            return x+1
    return 200


def lengthToConverge(method) :
    for x in range(1000) :
        result = method()
        if (result == 200) :
            return x
    return 1000



def test() :
    steps = 100
    res = [lengthToConverge(guess) for x in range(steps)]
    print ("Guessing Result Time to Converge " + str(np.mean(res)))
    res1 = [lengthToConverge(descend) for x in range(steps)]
    print ("Descent Result Time to Converge " + str(np.mean(res1)))

    plot.subplot(211)
    plot.hist(res)
    plot.title('Historgram of Guess Search - Mean ' + str(np.mean(res)))
    plot.subplot(212)
    plot.hist(res1)
    plot.title('Histogram of Descent Search - Mean ' + str(np.mean(res1)))
    plot.show()

test()
