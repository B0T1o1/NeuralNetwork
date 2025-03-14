

from math import e

def tanh(z):
    return (e**z - e**-z)/(e**z + e**-z)



class ActivationFunction:
    def __init__(self):
        pass
    def Activation_function(self):
        raise NotImplementedError
    def derivative_function(self):
        raise NotImplementedError



class Tanhshrink(ActivationFunction):
    def __init__(self):
        super().__init__()
        pass

    def Tanh_Shrink(self,z):
        x = z-tanh(z)
        return x
    
    def Activation_function(self,z):
        return self.Tanh_Shrink(z)
    
    def dxTanhShrink(self,z):
        x = tanh(z)**2
        return x
    
    def derivative_function(self,z):
        return self.dxTanhShrink(z)

class TanH(ActivationFunction):
    def __init__(self):
        super().__init__()
        pass
    def Activation_funcion(self,z):
        return tanh(z)
    def dxTanh(self,z):
        return 1 - tanh(z)**2
    def derivative_function(self,z):
        return self.dxTanh(self,z)

    


class CostFunction:
    def __init__(self):
        pass
    def Cost_function(self):
        raise NotImplementedError
    def derivative_function(self):
        raise NotImplementedError

class MeanSqauredError(CostFunction):
    def __init__(self):
        super().__init__()
        pass
    def Cost_function(self,Predicted,Actual):
        return (Predicted-Actual)**2
    def derivative_function(self,Predicted,Actual):
        return 2*(Predicted - Actual)

        
    

class Node:
    def __init__(self,InWeights:int,Activation_function:ActivationFunction):
        self._weights: list[float]= [0.0] * InWeights
        self.__activation_function = Activation_function
        self._biases:list[float] = [0.0] * InWeights
        self.lastz = 0
        self.lasta = 0
        


    def Node_forward(self,list_of_x):
        z = self.linear_combination(list_of_x)
        a = self.__activation_function.Activation_function(z)
        self.last_list_of_x = list_of_x
        self.lastz = z
        self.lasta = a
        return a
        


    def linear_combination(self,list_of_x):
        if len(list_of_x) == len(self._weights):
            total = 0
            for index, x in enumerate(list_of_x):
                total += self._weights[index] * x
                total +=  self._biases
            return total
        else:
            raise Exception('Number Of Arguments does not match expected Number of Inputs')
    

class NeuralNetwork:
    def __init__(self,Activation_function:ActivationFunction,Cost_function:CostFunction,LearningRate,NumberOfInputs:int, *args):
        self.__Cost_function = Cost_function
        self.__Activation_function = Activation_function
        self.__Expectedinps = NumberOfInputs
        self.__learningRate = LearningRate
        
        first_layer = []
        for i in range(args[0]):
            first_layer.append(Node(NumberOfInputs,self.__Activation_function))
        self.__Layers:list[list[Node]] = [first_layer]
        for layer in range(1,len(args)):
            Current_layer = []
            for Add_node in range(args[layer]):
                Current_layer.append(Node(args[layer-1],self.__Activation_function))
            self.__Layers.append(Current_layer)

    

    def forward(self,inputs:list[int]):
        if self.__Expectedinps == len(inputs):
            
            firstlayer = self.__Layers[0]
            results = []
            for i,n in enumerate(firstlayer):
                results.append(n.Node_forward(inputs))

            for ilayer in range(1,len(self.__Layers)):
                layer_results = []
                for node in self.__Layers[ilayer]:

                    layer_results.append(node.Node_forward(*results))
                results = layer_results
            return results[-1]
        else: raise Exception('Number Of Arguments does not match expected Number of Inputs') 

    def trainINP(self,inputs,y,m):
        if type(y) == int:
            self.forward(inputs)
            for i in range(len(self.__Layers),-1,-1):
                for node in self.__Layers[i]:
                    for iweight in range(len(node.last_list_of_x)):
                        node._weights[iweight] = node._weights[iweight] - (self.__learningRate/m)*( node.last_list_of_x[iweight] * self.__Cost_function.derivative_function(node.lasta , y) * self.__Activation_function.derivative_function(node.lastz))
                        node._biases[iweight] = node._biases[iweight] - (self.__learningRate/m)*(self.__Cost_function.derivative_function(node.lasta , y) * self.__Activation_function.derivative_function(node.lastz))
            return
        if type(y) == list:
            for iy in range(len(y)):
                self.forward(inputs)
            for i in range(len(self.__Layers),-1,-1):
                for node in self.__Layers[i]:
                    for iweight in range(len(node.last_list_of_x)):
                        node._weights[iweight] = node._weights[iweight] - (self.__learningRate/m)*( node.last_list_of_x[iweight] * self.__Cost_function.derivative_function(node.lasta , y[iy]) * self.__Activation_function.derivative_function(node.lastz))
                        node._biases[iweight] = node._biases[iweight] - (self.__learningRate/m)*(self.__Cost_function.derivative_function(node.lasta , y[iy]) * self.__Activation_function.derivative_function(node.lastz))
            return

    def trainStep(self,X,Y):
        m = len(X)
        for data in range(m):
            self.trainINP(X[data],Y[data],m)

    def train(self,X,Y,epochs):
        for i in range(epochs):
            self.trainStep(X,Y)



