# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 19:11:59 2020

@author: ArNo1
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:05:29 2019

@author: ArNo1
"""


"""Node object that can be use to build regression trees.

    Attributes
    ----------
    xVal: numpy.1darray
        The ordered x values for the regression tree starting from this node
    yVal: numpy.1darray
        Output value ordered according to increasing x values
    yMean: float
        The mean of y values for this node
    xCut: float or None
        Cut coordinate of the node, if it is not terminal, else None
    left : Node
        Left node, used to evaluate x values strictly lower than xCut
    right : Node
        Right node, used to evaluate x values higher or equal to xCut

        Parameters
        ----------
        xVal: numpy.1darray
            The x values for the regression tree starting from this node
            (must be ordered)
        yVal: numpy.1darray
            Output values ordered according to increasing x values
            (same size as x)
        """

import numpy as np
import matplotlib as plt
import math
import random

##### function we evaluate ####

def function(x):
    return np.sin(x)

#### variables ####

samplesize = 76
if (samplesize % 2) != 0:
    raise ValueError('Samplesize not divisible by 2!')
lambdavar = 5


#### Data ####

xRand = np.sort(np.random.uniform(0, 2*np.pi, samplesize))
yRand = function(xRand)
yRandNoised = np.random.normal(0, 0.01, samplesize) + yRand

xErrors = []
minError = 0

#### auxiliary functions ####

#Calculates the square of the 2-norm
def squaredsum(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays not of same length")
    result = 0
    for i in range(len(array1)):
        result += (array1[i]-array2[i])**2
    return result

#calculcates the index of the minimal error left + right
def MSE(array1, array2):
    coord = 0
    n = len(array1)
    error = math.inf
    for i in range(1, n-1):
        currenterror = squaredsum(array1[:i], array2[:i]) + squaredsum(array1[i:], array2[i:])
        if currenterror < error:
            coord = i
            error = currenterror
    return coord

#splits the 3 arrays into 6 random arrays of same length
def randomsplit(xarray, yarray, yNoisedarray):
    leftxarray = []
    leftyarray = []
    leftyNoisedarray = []
    l = len(xarray)
    k = l//2
    indices = random.sample(range(l),k) 
    indices = np.sort(indices)
    for entry in indices:
        leftxarray.append(xarray[entry])
        leftyarray.append(yarray[entry])
        leftyNoisedarray.append(yNoisedarray[entry])
    for entry in indices[::-1]:
        xarray = np.delete(xarray, entry)
        yarray = np.delete(yarray, entry)
        yNoisedarray = np.delete(yNoisedarray, entry)
    return xarray, yarray, yNoisedarray, leftxarray, leftyarray, leftyNoisedarray

#Finds the ordered place of xValue in xArray
def FindMiddleIndex(xValue, xArray):
    if xValue < xArray[0]:
        return 0
    if xValue > xArray[len(xArray) - 1]:
        return len(xArray) - 1
    i = 0
    while xValue > xArray[i] :
        i+=1
    return i

#predicts newYValues by linear interpolation of newXValues over yValues
def predict(xOriginal, yValues, newXValues):
    newYValues = []
    for entry in newXValues:
        index = FindMiddleIndex(entry, xOriginal)
        if index == 0:
            newYValues.append(yValues[0])
        elif index == len(xOriginal) - 1:
            newYValues.append(yValues[len(xOriginal) - 1])
        else:
            slope = (yValues[index] - yValues[index-1]) / (xOriginal[index] - xOriginal[index-1])
            newYValues.append(yValues[index-1] + slope*(entry - xOriginal[index-1]))
    return newYValues


#### Node ####

class Node(object):

    def __init__(self, xVal, yVal, yNoised, size, lam, minError, yFinal, depth):
        
        # Initial check on xVal
        if not np.all(np.sort(xVal) == xVal):
            raise ValueError('xVal must be ordered')
        if np.size(xVal) != np.size(yVal):
            raise ValueError('xVal and yVal must have the same size')
        if np.size(xVal) == 0:
            raise ValueError('Node should contain at least one data point')
          
        # Base attributes
        self.xVal = xVal
        self.yVal = yVal
        self.yNoised = yNoised
        self.yMean = np.mean(yVal)  # mean is computed once for all
        self.size = size
        self.lam = lam
        self.minError = minError
        self.yFinal = yFinal
        self.depth = depth
        
        # Attributes for non terminal nodes
        self.xCut = None
        self.left = None
        self.right = None
        
        self.develop()
        self.calcsize()
        
    @property
    
    def terminal(self):
        """Wether or not the node is terminal"""
        if (len(self.xVal) < self.lam):
            return True
        elif self.pruneable():
            return True
        else:
            return False
    
    def optimCut(self):
        "Sets the MSE coordinate"
        self.xCut = MSE(self.yVal, self.yNoised)

    def Cut(self, xCut):
        "Cuts the Node at xCut"
        if xCut == 0 or xCut == len(self.xVal):
            pass
        else:
            self.left = Node(self.xVal[:self.xCut], self.yVal[:self.xCut], self.yNoised[:self.xCut], self.size, self.lam, [0, self.minError[1]], self.yFinal, None)
            self.right = Node(self.xVal[self.xCut:], self.yVal[self.xCut:], self.yNoised[self.xCut:], self.size, self.lam, [0, self.minError[1]], self.yFinal, None)
            
    def develop(self):
        if self.terminal:
            self.yFinal.extend([np.mean(self.yNoised)]*len(self.xVal))
            self.minError[0] = math.inf
        else:
            self.optimCut()
            self.Cut(self.xCut)
            self.minError[0] = min(self.left.minError[0], self.right.minError[0], self.approxError())

            
    def calcsize(self):
        """Calculates size of the Node"""
        if self.terminal:
            self.size = 1
            self.depth = 1
        else:
            self.size = self.left.size + self.right.size
            self.depth = max(self.left.depth, self.right.depth) + 1
            
    def pruneable(self):
        if (self.approxError() <= self.minError[1]):
            return True
        return False
    
    def approxError(self):
        return squaredsum(self.yVal, [np.mean(self.yNoised)]*len(self.xVal))

    def Collapse(self):
        self.left = None
        self.right = None
        self.xCut = None

    def __call__(self, x, out=None):
        """Evaluate the tree on x (scalar or numpy vector), and store it
        into the out vector (if given)"""
        out = []
        return out

    def copy(self, xVal=None, yVal=None):
        """Copy the node into a new object similar as the original one,
        but with eventual child nodes different of those of the original"""
        if yVal is None:
            yVal = np.copy(self.yVal)
        if xVal is None:
            xVal = np.copy(self.xVal)
        c = Node(xVal, yVal)
        if not self.terminal:
            c.xCut = self.xCut
            iCut = np.argmin(xVal < c.xCut)
            c.left = self.left.copy(xVal[:iCut], yVal[:iCut])
            c.right = self.right.copy(xVal[iCut:], yVal[iCut:])
        return c

    def __str__(self):
        """Convert the node into a string that describes it"""
        if self.terminal:
            s = 'Node (term., yMean={:1.3f}, nVal={}, x in [{:1.3f}, {:1.3f}])'
            s = s.format(
                self.yMean, self.yVal, self.xVal.min(), self.xVal.max())
        else:
            s = 'Node (xCut={:1.3f}'.format(self.xCut)
            s += ', left=(mean={:1.3f}, nVal={}{})'.format(
                self.left.yMean, self.left.yVal,
                ', term.' if self.left.terminal else '')
            s += ', right=(mean={:1.3f}, nVal={}{})'.format(
                self.right.yMean, self.right.yVal,
                ', term.' if self.right.terminal else '')
            s += ')'
        return s
    
    def __repr__(self):
        """Method used for ipython terminal representation"""
        return self.__str__()  # equivalent to : return str(self)

    #def __getitem__(self, s):
        """Walk trough the tree to get a given node, using a path described
        by combination of letters ('r' for right and 'l' for left)
        given in the string s.

        Let T be a (non terminal) node. To get the left node of T, one writes

        >>> T['l']

        which is equivalent to

        >>> T.left

        To get the right node of the left node of T, one writes

        >>> T['lr']

        which is equivalent to

        >>> T.left.right

        Any ordered combination can be used, if there is a corresponding node.
        If the node does not exists, or some character in s does not correspond
        to left or right, then an error will be raised.
        Finally, empty string for s will simply return T itself.
        """
        """
        side = s[:1].lower()
        if side != '' and self.terminal:
            raise ValueError('cannot go further than terminal node')
        if side == 'r':
            return self.right[s[1:]]
        elif side == 'l':
            return self.left[s[1:]]
        elif side == '':
            return self
        else:
            raise ValueError(f'wrong character in combination : {side}')"""

### functions that directly operate using the Node ###

x = Node(xRand, yRand, yRandNoised, 0, lambdavar, [0, 0], [], None)

def prune(x, times, lam):
    Tree = [x]
    ComplexityList = []
    complexity = squaredsum(x.yVal, x.yFinal) + lam*x.size
    ComplexityList.append(complexity)
    '''
    print("##### ORIGINAL TREE: #####")
    print("##########################")
    print("Original minErr: ", x.minError[0])
    print("Original Size of the Tree: " , x.size)
    print("Original Predictive Error of the Tree: ", squaredsum(x.yVal, x.yFinal))
    print("Original Complexity of the Tree: " , complexity)
    print("Original Depth of the Tree: " , x.depth)
    plt.pyplot.plot(x.xVal, function(x.xVal), color = 'black')
    plt.pyplot.plot(x.xVal, x.yNoised, color = 'red')
    plt.pyplot.plot(x.xVal, x.yFinal, c = "green", linewidth = 2)
    plt.pyplot.figtext(0.75, 0.95, "Original function", fontsize = 11, c="orange")
    plt.pyplot.figtext(0.75, 0.925, "Noised function", fontsize = 11, c="red")
    plt.pyplot.figtext(0.75, 0.9, "Node(s)", fontsize = 11, c="black")
    plt.pyplot.figtext(0.85, 0.9, "Largest Node", fontsize = 11, c="green")
    '''
    for i in range(times):
        minError = x.minError[0]
        x = Node(x.xVal ,x.yVal, x.yNoised, 0, lambdavar, [0, minError], [], None)
        complexity = squaredsum(x.yVal, x.yFinal) + lam*x.size
        ComplexityList.append(complexity)
        '''
        print("##### Calculating new Node #####")
        print("##########################")
        print("new minErr: ", x.minError[0])
        print("Size of the Tree: " , x.size)
        print("Error of the Tree: ", squaredsum(x.yVal, x.yFinal))
        print("Depth of the Tree: " , x.depth)
        print("Complexity of the Tree: " , complexity)
        plt.pyplot.plot(x.xVal, x.yFinal, c = "black", alpha = (0.5*times+0.5))
        '''
        Tree.append(x)
        if x.minError[0] == math.inf:
            #print("Only terminal Nodes")
            break
    '''
    plt.pyplot.show()
    '''
    minComplexityNode = Tree[ComplexityList.index(min(ComplexityList))]
    return ComplexityList, Tree, minComplexityNode

'''
### Complexity graph: ###
ComplexityList = prune(x, 30, 0.001)[0]
plt.pyplot.plot(ComplexityList)
'''

#plt.pyplot.plot(xRand, yRand, color = 'red')

### 2-fold Cross-Validation score (plots the 2 predictions in blue&red and the original estimates in black)
def crossvalidation(DataList, lam):
        x1 = Node(DataList[0], DataList[1], DataList[2], 0, lambdavar, [0, 0], [], None)
        x2 = Node(DataList[3], DataList[4], DataList[5], 0, lambdavar, [0, 0], [], None)
        
        ### We now have 2 Nodes with lowest Complexity Criterion
        prediction1 = predict(x1.xVal, x1.yFinal, x2.xVal)
        prediction2 = predict(x2.xVal, x2.yFinal, x1.xVal) 
        score = squaredsum(x2.yFinal, prediction1) + squaredsum(x1.yFinal, prediction2)
        '''
        plt.pyplot.plot(x1.xVal, x1.yFinal, color = 'black')
        plt.pyplot.plot(x2.xVal, x2.yFinal, color = 'black')
        plt.pyplot.plot(x2.xVal, prediction1, color = 'blue')
        plt.pyplot.plot(x1.xVal, prediction2, color = 'orange')
        '''
        #print("Lambda = " + str(lam) + " has a prediction score of ", score)
        return score

### Calcule les prediction scores pour tout lambda dans ]0,1] avec accuracy
def lambdapredict(xRand, yRand, yRandNoised, accuracy):
    scores = []
    x = Node(xRand, yRand, yRandNoised, 0, lambdavar, [0, 0], [], None)
    for i in range(accuracy):   #for i in range(accuracy)
        xPruned = prune(x, x.size, (i+1)*(1/accuracy))[2] #First we get the node with least complexity for this lambda
        DataList = randomsplit(xPruned.xVal, xPruned.yVal, xPruned.yFinal)
        score = crossvalidation(DataList, (i+1)*(1/accuracy)) #The we calculate the predictive score of this lambda on the minNode
        scores.append(score)
    lambdamin = (scores.index(min(scores))+1)/accuracy
    #print("Best lambda: ", (lambdamin+1)*(1/accuracy), " with a prediction error of ", scores[lambdamin])
    return scores, lambdamin


scoresfinal = []

'''
### Drawing the scores of the different lambda 10 times, lambdaprecision = 10
for j in range(5):
    scoresfinal.append(lambdapredict(xRand, yRand, yRandNoised, 100)[0])
    
def draw(scoresfinal):
    x = [i for i in range(len(scoresfinal[0]))]
    for entry in scoresfinal:
        print(entry.index(min(entry)))
        plt.pyplot.scatter(x, entry)
        
draw(scoresfinal)
'''


### Prediction using random split
'''
DataList = randomsplit(xRand, yRand, yRandNoised)
prediction1 = predict(DataList[0], DataList[2], DataList[3])
prediction2 = predict(DataList[3], DataList[5], DataList[0])
plt.pyplot.plot(xRand, function(xRand), color = 'black')
plt.pyplot.plot(DataList[3], prediction1, color = 'red')
plt.pyplot.plot(DataList[0], prediction2, color = 'blue')
'''


### Prediction using odd/even split
'''
def split(array1, array2):
    array11 = []
    array12 = []
    array21 = []
    array22 = []
    for i in range(len(array1)):
        if i % 2 == 0:
            array11.append(array1[i])
            array21.append(array2[i])
        else:
            array12.append(array1[i])
            array22.append(array2[i])
    return array11, array12, array21, array22

def oddcrossvalidation(xRand, yRand, yRandNoised, lam):
    plt.pyplot.plot(xRand, yRand, color = 'black')
    xPruned = prune(x, x.size, lam)[2]
    DataList = split(xPruned.xVal, xPruned.yFinal)
    x1 = Node(DataList[0], DataList[2], DataList[2], 0, lambdavar, [0, 0], [], None)
    x2 = Node(DataList[1], DataList[2], DataList[3], 0, lambdavar, [0, 0], [], None)
    print("Remaining Sizes :", x1.size, x2.size)
    ### We now have 2 Nodes with lowest Complexity Criterion
    prediction1 = predict(x1.xVal, x1.yFinal, x2.xVal)
    prediction2 = predict(x2.xVal, x2.yFinal, x1.xVal) 
    score = squaredsum(x1.yNoised, prediction1) + squaredsum(x2.yNoised, prediction2)
    #plt.pyplot.plot(x1.xVal, x1.yFinal, color = 'black')
    #plt.pyplot.plot(x2.xVal, x2.yFinal, color = 'black')
    plt.pyplot.plot(x2.xVal, prediction1, color = 'blue')
    plt.pyplot.plot(x1.xVal, prediction2, color = 'orange')
    print("Lambda = " + str(lam) + " has a prediction score of ", score)
    return score

oddcrossvalidation(xRand, yRand, yRandNoised, 0.6)
'''

'''
DataList = [split(xRand)[0], split(xRand)[1], split(yRandNoised)[0], split(yRandNoised)[1]]
prediction1 = predict(DataList[0], DataList[2], DataList[1])
prediction2 = predict(DataList[1], DataList[3], DataList[0])

plt.pyplot.plot(xRand, function(xRand), color = 'black')
plt.pyplot.plot(DataList[1], prediction1, color = 'red')
plt.pyplot.plot(DataList[0], prediction2, color = 'blue')
'''

def average(xData, yData):
    print("AVERAGING")
    if len(xData) != len(yData):
        raise ValueError("Data not matching")
    length = 0
    for item in xData:
        length += len(item)
    newX = [i*((np.pi*2)/length) for i in range(length)]
    newY = []
    for value in newX:
        indexes = []
        y = 0
        for i in range(len(xData)):
            indexes.append(FindMiddleIndex(value, xData[i]))
        for i in range(len(indexes)):
            index = indexes[i]
            slope = (yData[i][index] - yData[i][index-1])/(xData[i][index] - xData[i][index-1])
            y += yData[i][index-1] + slope*(value - xData[i][index-1])
        newY.append(y)
    plt.pyplot.plot(newX, newY, color = 'pink')
    print(len(newX))
    print(newX)
### Tree

class RandForest(object):

    def __init__(self, size):
        
        # Base attributes
        self.size = size
        self.Trees = []
        self.xData = []
        self.yData = []
        
        self.grow()
        self.drawTree()
    
    def grow(self):
        for i in range(self.size):
            xRand = np.sort(np.random.uniform(0, 2*np.pi, samplesize))
            yRand = function(xRand)
            yRandNoised = np.random.normal(0, 0.01, samplesize) + yRand
            lam = lambdapredict(xRand, yRand, yRandNoised, 10)[1]
            xPruned = Node(xRand, yRand, yRandNoised, 0, lambdavar, [0, lam], [], None)
            self.Trees.append(xPruned)
            print("Lambda of Tree Number ", i+1, ": ",lam)
            print("Size of Tree Number", i+1, ": ", xPruned.size)
            
    def drawTree(self):
        for node in self.Trees:
            self.xData.append(node.xVal)
            self.yData.append(node.yFinal)
            #plt.pyplot.plot(node.xVal, node.yFinal)
        
               
Forest = RandForest(10)
average(Forest.xData, Forest.yData)

