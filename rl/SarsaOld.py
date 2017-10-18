import gym
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf

from utils import Utils, SimpleNetwork as simple

env = gym.make('CartPole-v0')


siz = 10
dimensions = [[4,siz],[siz,1]]

dimensions = [[5,siz],[siz,1]]
network = simple.SimpleNetwork('simple2',dimensions,.05)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.summary.SessionLog


def getAction(model,observation, search=.1) :
    obs = observation
    res1 = np.insert(obs,0,0.0).reshape(1,5)
    res2 = np.insert(obs,0,1.0).reshape(1,5)
    input = np.concatenate((res1,res2),0)
    exp = sess.run(network.out,{network.state:input})

    r_value = .25*np.random.randn(1)


    #if (exp[0] + r_value > exp[1]) :
    #    action =  0
    #else :
    #    action =  1

    action = Utils.e_greedy_uniform(.4, exp[0], exp[1])

    #r_value = np.random.rand(1)
    #if (r_value < search) :
    #    if (action == 0) : action = 1
    #    else : action = 0

    return action


def combineHistory(item, it) :
    rr = np.insert(item[it],0,item['a'])
    return rr

def monteCarloSarsa(network, history, complete = False, gain = .9) :
    hlen = len(history)
    max_value = 1 / (1 - gain)

    # Create the state action pairs for this case
    states = [combineHistory(x,'n') for x in history]
    exp = sess.run(network.out, {network.state: states})

    # Create the Value
    value = [gain*exp[x] + history[x]['r']/max_value for x in range(hlen)]
    if (not complete) : value[-1] = 0
    vstate = np.asarray(value).reshape((len(value),1))

    # Get the Q Values for Last State
    states = [combineHistory(x,'l') for x in history]

    ret = sess.run([network.internals(),network.optimize],{network.state:states, network.expect:vstate})
    err =  (np.mean(ret[0]['e']))
    return err


last = .5

def guess(last = .9):
    lstate = env.reset()
    model = np.random.randn(4)
    history = []
    error_history  = []
    error_history1 = []
    for x in range(250):
        #env.render()
        act = getAction(model,lstate,last)
        nstate, reward, done, info = env.step(act)
        history.append({'l':lstate,'n':nstate,'r':reward, 'a':act})
        lstate = nstate

        if (done) :
            if (x == 199) :
                complete = True
            else :
                complete = False
            err = monteCarloSarsa(network, history, complete)
            return [x,err]
            #err1 = td(network1,history, complete)
            #error_history.append(err)
            #error_history1.append(err1)
            #return [x+1,error_history, error_history1]

    #err = monteCarlo(network,history, True)
    #err1 = td(network1,history, True)
    #error_history.append(err)
    #return [200, error_history]






def test() :
    res = []
    last = .25
    mval = 8
    for x in range(500) :
        result = guess(last)
        if (result[0] > mval) :
            mval = result[0]
        res.append(result)
        if (result[0] >= 120) :
            last = .1
        else :
            last = .2
        #last = .25 - .25*result[0]/200
    #res = [guess() for x in range(1000)]
    #print ("Test1 " + str(np.mean(res[0])))
    tim  = Utils.flatten(res, 0)
    err  = Utils.flatten(res, 1)

    plot.subplot(211)
    plot.plot(np.asarray(err))
    plot.subplot(212)
    plot.plot(np.asarray(tim))
    plot.show()

#test()
