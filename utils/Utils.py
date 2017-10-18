import numpy as np

def flatten(input,name) :
    return [x[name] for x in input]

def geom(alpha,length) :
    return (1-pow(alpha,length))/(1-alpha)

def e_greedy(range,p1,p2, uniform=True) :
    if (uniform) :
        r_value = 2 * range * (np.random.rand(1) - .5)
    else :
        r_value = range * (np.random.randn(1))

    value = p1 - p2 + r_value
    if (value > 0):
        val = 0
    else:
        val = 1
    return val

def e_greedy_uniform(range, p1, p2) :
    return e_greedy(range,p1,p2,True)

def e_greedy_normal(range, p1, p2) :
    return e_greedy(range,p1,p2,False)