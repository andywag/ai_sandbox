"""
Module containing utilities for dealing with one hot encoding of information
"""
class Group:
    """A group of items using one-hot encoding"""
    def __init__(self,inputs):
        self.items = Item.items(inputs)
    def siz(self):
        return len(self.items)
    def empty(self):
        return Item.empty('empty', self.siz())
    def last(self):
        return self.items[-1]


    @staticmethod
    def array(name, len):
        items = ["name_" + str(x) for x in range(len)]
        return Group(items)

class Item :
    """An indiviual item using one hot encoding. """
    def __init__(self,item,index, siz) :
        self.item = item
        self.index = index
        self.siz   = siz
        self.arr   = self.__create__()
    def __create__(self):
        def cr(x) :
            if (x == self.index) : return 1.0
            else : return 0.0
        arr = [cr(x) for x in range(self.siz)]
        return arr
    def __str__(self):
        return self.item + "(" + str(self.index) + ")"

    def empty(self):
        return Item.empty('empty',self.siz)

    @staticmethod
    def items(inputs):
        return [Item(o, i, len(inputs)) for i,o in enumerate(inputs)]

    @staticmethod
    def empty(name, siz):
        return Item(name, siz+1, siz)