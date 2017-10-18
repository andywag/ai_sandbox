import gym
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf

from utils import Utils, SimpleNetwork as simple

env = gym.make('CartPole-v0')

def getAction(model,observation) :
    action = 0 if np.matmul(model, observation) < 0 else 1
    return action


siz = 7
dimensions = [[4,siz],[siz,1]]

network  = simple.SimpleNetwork('simple',dimensions,.1)
network1 = simple.SimpleNetwork('simple1',dimensions,.04)

dimensions = [[5,siz],[siz,1]]
network2 = simple.SimpleNetwork('simple2',dimensions,.04)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.summary.SessionLog

def monteCarloQLearn(network, history, complete = False, gain = .9) :
    hlen = len(history)
    max_value = 1/(1-gain)
    if (complete) :
        value = [Utils.geom(gain, 200) / max_value for x in range(hlen)]
    else :
        value = [Utils.geom(gain, hlen - x) / max_value for x in range(hlen)]

    istate = np.asarray(Utils.flatten(history, 'l'))
    vstate = np.asarray(value).reshape((len(value),1))

    ret = sess.run([network.internals(),network.optimize],{network.state:istate, network.expect:vstate})
    err =  (np.mean(ret[0]['e']))
    return err

def updateValue(lstate,nstate, reward, act) :
    #print(lstate, nstate, reward, act)
    n1 = nstate.reshape([1,4])
    o1 = np.ndarray([1]).reshape([1,1])
    [nint,nValue] = sess.run([network.internals(),network.out], {network.state:n1,network.expect:o1})
    eValue        = reward/200 + .9*nValue
    l1 = lstate.reshape([1,4])
    o2 = np.asarray([eValue]).reshape([1,1])
    np.ndarray
    cval = sess.run([network.internals(), network.optimize],{network.state:l1,network.expect:o2})
    print (eValue, cval[0]['e'], cval[0]['o'])


def monteCarlo(network,history, complete = False, gain = .9) :
    hlen = len(history)
    max_value = 1/(1-gain)
    if (complete) :
        value = [Utils.geom(gain, 200) / max_value for x in range(hlen)]
    else :
        value = [Utils.geom(gain, hlen - x) / max_value for x in range(hlen)]

    istate = np.asarray(Utils.flatten(history, 'l'))
    vstate = np.asarray(value).reshape((len(value),1))

    ret = sess.run([network.internals(),network.optimize],{network.state:istate, network.expect:vstate})
    err =  (np.mean(ret[0]['e']))
    return err


def td(network,history, complete = False, gain = .9) :
    hlen = len(history)
    max_value = 1/(1-gain)
    # Calculate the Expected Value Function Based on the State Input
    exp = sess.run(network.out, {network.state:np.asarray(Utils.flatten(history, 'n'))})

    value = [gain*exp[x] + max_value/200 for x in range(hlen)]

    istate = np.asarray(Utils.flatten(history, 'l'))
    vstate = np.asarray(value).reshape((len(value),1))

    ret = sess.run([network.internals(),network.optimize],{network.state:istate, network.expect:vstate})
    err =  (np.mean(ret[0]['e']))
    return err

def guess():
    lstate = env.reset()
    model = np.random.randn(4)
    history = []
    error_history  = []
    error_history1 = []
    for x in range(250):
        #env.render()
        act = getAction(model,lstate)
        nstate, reward, done, info = env.step(act)
        history.append({'l':lstate,'n':nstate,'r':reward, 'a':act})
        lstate = nstate

        if (done) :
            if (x == 199) :
                complete = True
            else :
                complete = False
            err = monteCarlo(network, history, complete)
            err1 = td(network1,history, complete)
            error_history.append(err)
            error_history1.append(err1)
            return [x+1,error_history, error_history1]

    #err = monteCarlo(network,history, True)
    #err1 = td(network1,history, True)
    #error_history.append(err)
    #return [200, error_history]






def test() :
    res = [guess() for x in range(2000)]
    #print ("Test1 " + str(np.mean(res[0])))
    err  = Utils.flatten(res, 1)
    err1 = Utils.flatten(res, 2)
    plot.subplot(211)
    plot.plot(np.asarray(err))
    plot.subplot(212)
    plot.plot(np.asarray(err1))
    plot.show()

#test()
