
"""
Module containing learning algorithms for general use. Currently only supports a block version of
Q-Learning and SARSA. The next step will be to change these to sample based updates and add different
algorithms
"""

import numpy as np
import tensorflow as tf

from rl.StateUtil import ActionHistory
from utils import Utils, SimpleNetwork as simple


class LearnBase:
    def __init__(self, dimensions):
        self.createBase(dimensions)

    def createBase(self,dimensions):
        network = simple.SimpleNetwork('simple2', dimensions, .1)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        tf.summary.SessionLog

        self.network = network
        self.sess = sess

    def resetSession(self):
        #tf.Session.reset(self.sess)
        self.sess.run(tf.global_variables_initializer())


class BlockQLearn(LearnBase):
    def __init__(self, dimensions):
        super(BlockQLearn,self).__init__(dimensions)


    def action(self,history, range=.1):
        observation = history.lastState()
        # Append both actions to the observation to create
        res1 = np.insert(observation, 0, 0.0).reshape(1, observation.shape[0] + 1)
        res2 = np.insert(observation, 0, 1.0).reshape(1, observation.shape[0] + 1)
        input = np.concatenate((res1, res2), 0)
        # Run the network to determine the best action given the current state
        exp = self.sess.run(self.network.out, {self.network.state: input})
        # Run action based on e_greedy policy
        action = Utils.e_greedy_uniform(range, exp[0], exp[1])
        return action

    def learn(self, act_history: ActionHistory, complete=False, gain=.9):

        hlen = len(act_history.history)
        max_value = 1 / (1 - gain)
        if (complete):
            value = [Utils.geom(gain, 200) / max_value for x in range(hlen)]
        else:
            value = [Utils.geom(gain, hlen - x) / max_value for x in range(hlen)]
        vstate = np.asarray(value).reshape((len(value), 1))

        states = np.asarray([np.insert(item.last, 0, item.action) for item in act_history.history])

        ret = self.sess.run([self.network.internals(), self.network.optimize], {self.network.state: states, self.network.expect: vstate})
        err = (np.mean(ret[0]['e']))
        return err

class BlockSarsa(LearnBase) :
    def __init__(self, dimensions):
        super(BlockSarsa,self).__init__(dimensions)

    def action(self,history, range=.1):
        obs = history.lastState()
        res1 = np.insert(obs, 0, 0.0).reshape(1, 5)
        res2 = np.insert(obs, 0, 1.0).reshape(1, 5)
        input = np.concatenate((res1, res2), 0)
        exp = self.sess.run(self.network.out, {self.network.state: input})

        action = Utils.e_greedy_uniform(.4, exp[0], exp[1])

        return action

    def combineHistory(self,input, action):
        rr = np.insert(input, 0, action)
        return rr

    def learn(self, act_history: ActionHistory, complete=False, gain=.9):
        history = act_history.history
        hlen = len(history)
        max_value = 1 / (1 - gain)

        # Create the state action pairs for this case
        states = [self.combineHistory(x.next, x.action) for x in history]
        exp = self.sess.run(self.network.out, {self.network.state: states})

        # Create the Value
        value = [gain * exp[x] + history[x].reward / max_value for x in range(hlen)]
        if (not complete): value[-1] = 0
        vstate = np.asarray(value).reshape((len(value), 1))

        # Get the Q Values for Last State
        states = [self.combineHistory(x.last, x.action) for x in history]

        ret = self.sess.run([self.network.internals(), self.network.optimize],
                            {self.network.state: states, self.network.expect: vstate})
        err = (np.mean(ret[0]['e']))
        return err


