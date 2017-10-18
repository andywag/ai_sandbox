"""
Module containing classes for generating N-stage simple neural networks
"""
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
import functools as ft

class TopNetwork :
    def __init__(self, name, dims, gain):
        self.name = name
        self.dims = dims
        self.gain = gain
        self.createVars()

    def createVars(self):
        """Create the variables used in the network """
        self.taps = []
        self.bias = []
        for idx, dim in enumerate(self.dims) :
            print (dim)
            self.taps.append(tf.get_variable(self.name + "_st_" + str(idx) + "_w",shape = dim))
            self.bias.append(tf.get_variable(self.name + "_st_" + str(idx) + "_b",shape = [1,dim[1]]))
        self.input = tf.get_variable(self.name + "_input",shape = [dim[0]])
        self.output = tf.get_variable(self.name + "_expect",shape = [1,dim[1]])


class SimpleNetwork(TopNetwork) :
    """ Simple neural network with arbitrary dimensions and numbers of stages.
    name : string : Name of Network
    dims : [[int,int]] : Dimensions of network specified in 2dimensional list of tuples
    gain : int         : Gain of network
    """
    def __init__(self, name, dims, gain):
        TopNetwork.__init__(self, name, dims, gain)
        self.state = tf.placeholder(tf.float32, shape = [None,dims[0][0]])
        self.createExpect()

        self.createLast()
        self.createError()


    def createExpect(self):
        self.expect = tf.placeholder(tf.float32, shape=[None, self.dims[-1][1]])

    def createVars(self):
        self.taps = []
        self.bias = []
        for idx, dim in enumerate(self.dims) :
            print (dim)
            self.taps.append(tf.get_variable(self.name + "_st_" + str(idx) + "_w",shape = dim))
            self.bias.append(tf.get_variable(self.name + "_st_" + str(idx) + "_b",shape = [1,dim[1]]))
        self.input = tf.get_variable(self.name + "_input",shape = [dim[0]])
        self.output = tf.get_variable(self.name + "_expect",shape = [1,dim[1]])


    def createLast(self):
        input = self.state
        for i in range(len(self.dims)) :
            w = self.taps[i]
            b = self.bias[i]
            input = tf.sigmoid(tf.add(tf.matmul(input, w), b))

        self.out = input

    def createError(self):
        self.loss = tf.nn.l2_loss(self.expect - self.out)
        self.rawError = tf.subtract(self.expect,self.out)
        self.error = tf.square(self.rawError)
        self.optimize = tf.train.AdamOptimizer(self.gain).minimize(self.error)


    def internals(self):
        return {'w' : self.taps, 'b' : self.bias, 'l' : self.loss, 'o' : self.out, 'ra' : self.rawError, 'e':self.error, 'ex':self.expect}


class PDA(TopNetwork) :
    """
    Special case of the simple network which allows for the error to be calculated with
    different feedback mechanism
    """
    def __init__(self,name, dims, gain, size):
        # Inputs to the neural network consist of input state, input symbol, input read
        self.size = size
        TopNetwork.__init__(self, name, dims, gain)
        self.state = tf.placeholder(tf.float32, shape = [None,size[0]])
        self.input = tf.placeholder(tf.float32, shape = [None,size[1]])
        self.read  = tf.placeholder(tf.float32, shape = [None,size[2]])

        self.createNetwork()
        self.createError()


    def createNetwork(self):

        input = tf.concat([self.state,self.input,self.read],1)
        for i in range(len(self.dims)) :
            w = self.taps[i]
            b = self.bias[i]
            input = tf.sigmoid(tf.add(tf.matmul(input, w), b))
        self.out = input
        self.state_out  = self.out[0][0:4]
        self.action_out = self.out[0][4]
        pass

    def createError(self):
        final_state = tf.constant([1.0 for x in range(self.size[2])])
        self.error  = tf.nn.l2_loss(tf.subtract(self.state_out,final_state))
        self.optimize = tf.train.AdamOptimizer(self.gain).minimize(self.error)
