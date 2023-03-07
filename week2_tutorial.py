import numpy as np

def finiteDiff(fn, x, parameters, delta, multiplier=1):
    shape = parameters.shape
    var = parameters.reshape(-1)
    diff = []
    for idx in range(len(var)):
        var[idx] += delta/2
        yplus = fn(x)
        var[idx] -= delta
        yminus = fn(x)
        varDiff = ((yplus - yminus) / delta * multiplier).sum()
        diff.append(varDiff)

        # restore
        var[idx] += delta/2
    return np.array(diff).reshape(*shape)

class Node:
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters
        self.parameters_deltas = [None for _ in range(len(self.parameters))]


class Linear(Node):
    def __init__(self, input_shape, output_shape, weight=None, bias=None):
        if weight is None:
            # naive
            weight = np.random.randn(input_shape, output_shape) * 0.01
            # xavier
            weight = np.random.randn(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape))
        if bias is None:
            bias = np.zeros(output_shape)
        super(Linear, self).__init__('linear', [weight, bias])

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.parameters[0]) + self.parameters[1]

    def backward(self, delta):
        self.parameters_deltas[0] = self.x.T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta, 0)
        return delta.dot(self.parameters[0].T)


class Sigmoid(Node):
    def __init__(self):
        super(Sigmoid, self).__init__('sigmoid', [])

    def forward(self, x, *args):
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return delta * ((1 - self.y) * self.y)


def net_forward(net, x):
    for node in net:
        x = node.forward(x)
    return x


def net_backward(net, y_delta):
    for node in net[::-1]:
        y_delta = node.backward(y_delta)
    return y_delta

if __name__ == "__main__":
    delta = 0.01
    precision = 1e-3

    weight = np.random.randn(5, 10) * 0.01
    bias = np.random.randn(10) * 0.01
    linear = Linear(10, 5, weight, bias)

    x = np.random.uniform(0.1, 1, [6, 5])

    weightFD = finiteDiff(linear.forward, x, weight, delta)
    biasFD = finiteDiff(linear.forward, x, bias, delta)
    xFD = finiteDiff(linear.forward, x, x, delta)

    y = linear.forward(x)
    xDiff = linear.backward(np.ones([6, 10]))

    sigmoid = Sigmoid()

    xFD = finiteDiff(sigmoid.forward, x, x, delta)

    y = sigmoid.forward(x)
    xDiff = sigmoid.backward(np.ones([6, 5]))

    graph = [linear, sigmoid]

    weightFD = finiteDiff(lambda x: net_forward(graph, x), x, weight, delta)
    biasFD = finiteDiff(lambda x: net_forward(graph, x), x, bias, delta)
    xFD = finiteDiff(lambda x: net_forward(graph, x), x, x, delta)

    y = net_forward(graph, x)
    xDiff = net_backward(graph, np.ones([6, 10]))

    import pdb
    pdb.set_trace()

    (xDiff - xFD)
    (weightFD-linear.parameters_deltas[0])
    (biasFD-linear.parameters_deltas[1])

