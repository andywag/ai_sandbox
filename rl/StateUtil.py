"""
Module which contains classes to help store the state of the system
"""

class Action:
    def __init__(self,lstate,nstate,reward,action,done, info):
        self.last = lstate
        self.next = nstate
        self.reward = reward
        self.action = action
        self.done   = done
        self.info   = info

class ActionHistory:
    def __init__(self,env):
        self.history = []
        self.env     = env
    def run(self,action):
        s,r,d,i = self.env.step(action)
        self.history.append(Action(self.lastState(),s,r,action,d, i))
    def lastState(self):
        if (len(self.history) == 0) :
            observation = self.env.reset();
            self.history.append(Action(observation,observation,0,0,0,0))
            return observation
        else:
            return self.history[-1].next
    def isDone(self):
        return self.history[-1].done

