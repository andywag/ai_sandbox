
import numpy as np
import tensorflow as tf

from utils import SimpleNetwork as simple
from utils import OneHot as oh


class StackItem :
    def __init__(self,item,depth):
        self.item  = item
        self.depth = depth

class ContinuousStack :
    def __init__(self, no_range = .1, typ = None):
        self.stack = []
        self.no_range = no_range
        self.typ = typ
    def clear(self):
        self.stack = []
    def push(self,item:StackItem):
        self.stack.append(item)
    def pop(self, depth):
        diff = depth
        while (diff > 0) :
            remainder = self.stack[0].depth - depth
            if (remainder < 0) :
                self.stack[0].depth = self.stack[0].depth - diff
                diff = 0
            else :
                self.stack.pop()
                diff = remainder
    def action(self,item, action):
        if (action > self.no_range) :
            self.push(StackItem(item,action))
        elif (action < -self.no_range) :
            self.pop(-self.no_range)
    def fill(self):
        depth = [item.depth for item in self.stack]
        return sum(depth)

    def topOneHot(self):
        dep = 0
        stat = self.typ.empty().arr
        for item in self.stack :
            if (dep < 1000) :
                stat[item.item.index] = stat[item.item.index] + item.depth
                dep = dep + item.depth
        return stat

    def top(self):
        if (len(self.stack) == 0) :
            if (self.typ is None) :
                return 0.0
            else :
                return self.typ.empty().arr
        elif (isinstance(self.typ,oh.Group)) :
            return self.topOneHot()
        else :
            return self.top()
    def debug(self):
        result = [str(item.item) + ","+ str(item.depth) for item in self.stack]
        return result

class NNPDA :
    def __init__(self, states, inputs, siz, gain):
        self.states = states
        self.inputs = inputs
        self.stack = ContinuousStack(typ = inputs)

        dim = [[12, siz], [siz, siz], [siz, 5]] # {State,Input,Read} -> {State,Action}

        self.nn    = simple.PDA('value',dim,gain,[4,4,4])

    def train(self, input):
        state = inputs.empty().arr # Intialize the input state to the beginning
        act = []
        for x in range(len(input)) :
            item = self.stack.top()
            input_vector = state + input[x].arr + item
            ninput = np.asarray(input_vector)
            res = sess.run([self.nn.out], {self.nn.state: [ninput]})
            t = res[0][0]
            state = t[0:len(state)].tolist()
            action = 2 * (t[len(state)] - .5)
            act.append(action)
            #print("Run ", state, action)
            self.stack.action(input[x], action)
            pass

        item = self.stack.top()
        input_vector = state + self.inputs.last().arr + item
        ninput = np.asarray(input_vector)
        expect = [1.0 for x in range(states.siz())]
        res = sess.run([self.nn.out, self.nn.optimize, self.nn.error, self.nn.fill],
            {self.nn.state: [ninput], self.nn.expect:[expect], self.nn.fill:[[self.stack.fill()]]})

        print (res[2],res[3],act,res[0], self.stack.debug())
        self.stack.clear()


            #sess.run()


[open,data,close,last] = oh.Item.items(['open','data','close','last'])
inputs = oh.Group(['open','data','close','last'])
states = oh.Group.array('state',4)





nnpda = NNPDA(states,inputs,20,.01)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.summary.SessionLog




vec = [[open, data, close],[open, data, data, close], [open, data,data,data,close]]

for x in range(100) :
    for y in range(4) :
        nnpda.train(vec[x % 1])

