class neuron:
    def __init__(self, weights, bias, inputs):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs

    def output(self):
        size = len(self.weights)
        ans = 0
        for i in range(size):
            ans += self.weights[i] * self.inputs[i]
        return ans + self.bias
    
n1 = neuron([1,2,3],5,[2,4,6])
n2 = neuron([2,5,8], 3.5,[1,4,3])

l1 = [n1,n2]
for i in l1:
    print("output:", i.output())