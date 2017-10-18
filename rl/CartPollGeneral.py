import gym
import matplotlib.pyplot as plot
import numpy as np

from rl import StateUtil as da, Training as tr
from utils import Utils

COMPLETE_RUNS = 200
COMPLETE_CHECK = COMPLETE_RUNS -1


def guess(algo,env,last = .9):
    act_history = da.ActionHistory(env)
    for x in range(COMPLETE_RUNS):
        # Retrieve the action from the model and run the state
        action = algo.action(act_history)
        act_history.run(action)
        if (act_history.isDone()) :
            if (x == COMPLETE_CHECK) : complete = True
            else :
                complete = False
                act_history.history[-1].reward = 0
            err = algo.learn(act_history,complete)
            return [x,err]


def single_test(algo,gain,plot_enable=True) :
    """
    Runs a single test
    :param plot_enable: Enables plotting
    :return: int : The length of time for the algorithm to converge
    """
    env = gym.make('CartPole-v0')
    res = []
    algo.resetSession()

    # Run the simulation n times to check teh convergence time
    for x in range(1000) :
        result = guess(algo,env,gain)
        res.append(result)
        if (result[0] == COMPLETE_CHECK) :
            #if (plot_enable) :
            print("Simulation Converged in " + str(x) + "runs")
            break

    if (plot_enable) :
        tim  = Utils.flatten(res, 0)
        err  = Utils.flatten(res, 1)

        plot.subplot(211)
        plot.plot(np.asarray(err))
        plot.subplot(212)
        plot.plot(np.asarray(tim))
        plot.show()

    return x

def converge_test(algo,gain) :
    """
    Runs a set of tests to check the average convergence time of the algorithm used
    """
    steps = 20
    res = [single_test(algo,gain,False) for x in range(steps)]
    print("Guessing Result Time to Converge " + str(np.mean(res)))

    plot.hist(res)
    plot.title('Historgram of Guess Search - Mean ' + str(np.mean(res)))
    plot.show()


siz = 10
dimensions = [[5,siz],[siz,1]]

q_learn      = tr.BlockQLearn(dimensions)
q_gain       = .5

#sarsa_learn  = tr.BlockSarsa(dimensions)
#sarsa_gain   = .9

algo = q_learn
gain = q_gain

#single_test(algo,gain)
converge_test(algo,gain)