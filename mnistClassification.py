import numpy as np
import subprocess
import os
import struct
import json


class Node:
    '''
    Node template for nodes in computational graph.

    Attributes:
        name (string): name for the node;
        parameters (list of ndarray): list used to save node's parameters;
        parameters_deltas (list of ndarray): list of gradients.
    '''
    def __init__(self, name, parameters=None):
        self.name = name    
        self.parameters = parameters
        self.parameters_deltas = [None for _ in range(len(self.parameters))]


class Linear(Node):
    '''
    Linear layer
    '''
    def __init__(self, input_shape, output_shape, weight=None, bias=None):
        '''
        Args:
            input_shape (int): input shape;
            output_shape (int): output shape;
            x (2darray): input array of shape (batch_size, num_pixels).
        '''
        if weight is None:
            # weight = np.random.randn(input_shape, output_shape) * 0.01
            '''
            Implement the Xavier initalization here
            '''
            # implement Xavier initialization here
            weight = np.random.randn(input_shape, output_shape) * np.sqrt(1 / input_shape)

        if bias is None:
            bias = np.zeros(output_shape)
        super(Linear, self).__init__('linear', [weight, bias])

    def forward(self, x):
        '''
        Args:
            x (2darray): input array of shape (batch_size, n_variables).

        Returns:
            ndarray: linear layer result.
        '''
        self.x = x
        return np.matmul(x, self.parameters[0]) + self.parameters[1]

    def backward(self, delta):
        '''
        Args:
            delta (ndarray): gradient of L with repect to node's output, dL/dy.

        Returns:
            ndarray: gradient of L with respect to node's input, dL/dx
        '''
        self.parameters_deltas[0] = self.x.T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta, 0)
        return delta.dot(self.parameters[0].T)


class Sigmoid(Node):
    '''
    Sigmoid activation function
    '''
    def __init__(self):
        super(Sigmoid, self).__init__('sigmoid', [])

    def forward(self, x, *args):
        '''
        Args:
            x (2darray): input array of shape (batch_size, n_variables).

        Returns:
            ndarray: sigmoid activation result.
        '''
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        '''
        Args:
            delta (ndarray): gradient of L with repect to node's output, dL/dy.

        Returns:
            ndarray: gradient of L with respect to node's input, dL/dx
        '''
        return delta * ((1 - self.y) * self.y)


class Softmax(Node):
    '''
    Softmax activation function
    '''
    def __init__(self):
        super(Softmax, self).__init__('softmax', [])

    def forward(self, x):
        '''
        Implement this part
        '''
        # implement softmax here
        if x.ndim == 2:
            x = x.T 
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            self.output = y.T
            return self.output
        

        x = x - np.max(x) # ????????????
        self.output = np.exp(x) / np.sum(np.exp(x))
        return self.output
     
    def backward(self, delta):
        '''
        Implement backward pass of softmax function

        Args:
            delta (ndarray): gradient of loss with respect to output of this node, of shape (batch_size, num_classes)

        Returns:
            d_inputs (ndarray): gradient of loss with respect to input of this node, of shape (batch_size, num_classes)
        '''
        y = self.output # output of softmax node from forward pass
        batch_size = y.shape[0]
        num_classes = y.shape[1]
        
        grads = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            # repeat y n times to form a 2d array
            y1 = np.tile(y[i], (num_classes, 1))
            # 5x5 identity matrix
            i = np.identity(num_classes)
            # repeat y n times to form a 2d array
            y2 = y1.T
            jacobian_matrix = y2 * (i - y1)      
            d_inputs = np.dot(delta[i], jacobian_matrix)
            grads[i] = d_inputs

        return grads
        

class CrossEntropy(Node):
    '''
    CrossEntropy loss function
    '''
    def __init__(self):
        super(CrossEntropy, self).__init__('cross_entropy', [])

    def forward(self, x, l):
        '''
        Implement this part
        '''
        if x.ndim == 1:
            l = l.reshape(1, l.size)
            x = x.reshape(1, x.size)
            
        # ???????????????one-hot-vector????????????????????????????????????????????????
        if l.size == x.size:
            l = l.argmax(axis=1)
                
        batch_size = x.shape[0]
        self.x = x
        self.l = l
        return -np.sum(np.log(x[np.arange(batch_size), l] + 1e-7))
    
    def backward(self, delta):
        '''
        Implement this part
        '''
        # implement cross entropy backward here
        batch_size = self.x.shape[0]
        grad = np.zeros_like(self.x)
        grad[np.arange(batch_size), self.l] = -1 / (self.x[np.arange(batch_size), self.l] + 1e-7)
        return delta * grad



class Mean(Node):
    '''
    Mean function
    '''
    def __init__(self):
        super(Mean, self).__init__('mean', [])

    def forward(self, x):
        '''
        Args:
            x (2darray): input array of shape (batch_size, n_variables).

        Returns:
            ndarray: mean function result.
        '''
        self.x = x
        return x.mean()

    def backward(self, delta):
        '''
        Args:
            delta (ndarray): gradient of L with repect to node's output, dL/dy.

        Returns:
            ndarray: gradient of L with respect to node's input, dL/dx
        '''
        return delta * np.ones(self.x.shape) / np.prod(self.x.shape)


def load_MNIST():
    '''
    Download and unpack MNIST data.

    Returns:
        tuple of ndarray: tuple of length 4. They are training set data, training set label,
            test set data and test set label.
    '''
    base = "http://yann.lecun.com/exdb/mnist/"
    objects = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
               'train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
    end = ".gz"
    path = "data/raw/"
    cmd = ["mkdir", "-p", path]
    subprocess.check_call(cmd)
    print('Downloading MNIST dataset. Please do not stop the program\
    during the download. If you do, remove `data` folder and try again.')
    for obj in objects:
        if not os.path.isfile(path + obj):
            cmd = ["wget", base + obj + end, "-P", path]
            subprocess.check_call(cmd)
            cmd = ["gzip", "-d", path + obj + end]
            subprocess.check_call(cmd)

    def unpack(filename):
        '''
        Unpack file.
        '''
        with open(filename, 'rb') as f:
            _, _, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))
                          [0] for d in range(dims))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
            return data

    # load objects
    data = []
    for name in objects:
        name = path + name
        data.append(unpack(name))
    labels = np.zeros([data[1].shape[0], 10])
    for i, iterm in enumerate(data[1]):
        labels[i][iterm] = 1
    data[1] = labels
    labels = np.zeros([data[3].shape[0], 10])
    for i, iterm in enumerate(data[3]):
        labels[i][iterm] = 1
    data[3] = labels
    return data


def random_draw(data, label, batch_size):
    '''
    Randomly draw.

    Args:
        data (ndarray): dataset of shape (batch_size, n_variables)
        label (ndarray): one-hot label for dataset,
            for example, 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        batch_size (int): size of batch.

    Returns:
        2darray: image data batch;
        2darray: label of images draw.
    '''
    perm = np.random.permutation(data.shape[0])
    data_b = data[perm[:batch_size]]
    label_b = label[perm[:batch_size]]
    return data_b.reshape([data_b.shape[0], -1]) / 255.0, label_b

def match_ratio(result, label):
    '''the ratio of result matching target.'''
    label_p = np.argmax(result, axis=1)
    label_t = np.argmax(label, axis=1)
    ratio = np.sum(label_p == label_t) / label_t.shape[0]
    return ratio


def net_forward(net, x, label):
    '''forward function for this sequencial network.'''
    for node in net:
        if node.name == 'cross_entropy':
            result = x
            x = node.forward(x, label)
        else:
            x = node.forward(x)
    return result, x


def net_backward(net):
    '''backward function for this sequencial network.'''
    y_delta = 1.0
    for node in net[::-1]:
        y_delta = node.backward(y_delta)
    return y_delta

if __name__ == '__main__':
    batch_size = 200
    learning_rate = 0.3
    dim_img = 784
    num_digit = 10
    # an epoch means running through the training set roughly once
    num_epoch = 10
    test_data, test_label, train_data, train_label = load_MNIST()
    num_iteration = len(train_data) // batch_size


    # define a list as a network, nodes are chained up
    net = [Linear(dim_img, num_digit), Softmax(), CrossEntropy(), Mean()]

    nparams = 0
    for term in net:
        for para in term.parameters:
            nparams += np.prod(para.shape)
    print('total nubmer of trainable parameters:', nparams)

    # display test loss before training
    x, label = random_draw(test_data, test_label, 1000)
    result, loss = net_forward(net, x, label)
    print('Before Training.\nTest loss = %.4f, correct rate = %.3f' % (loss, match_ratio(result, label)))

    for epoch in range(num_epoch):
        for j in range(num_iteration):
            x, label = random_draw(train_data, train_label, batch_size)
            result, loss = net_forward(net, x, label)

            net_backward(net)

            '''
            Implement the SGD update here
            '''
            raise Exception("No SGD implementation")

        # test set results
        result_test, loss_test = net_forward(net, test_data.reshape(test_data.shape[0], -1), test_label)
        print("epoch = %d/%d, loss = %.4f, corret rate = %.3f, test correct rate = %.3f" %
              (epoch, num_epoch, loss, match_ratio(result, label), match_ratio(result_test, test_label)))

    # final score report of the test set
    result_test, loss_test = net_forward(net, test_data.reshape(test_data.shape[0], -1), test_label)
    print('Test loss = %.4f, correct rate = %.3f' % (loss_test, match_ratio(result_test, test_label)))

    # saving the model
    layer_string = []
    layer_paramters = []
    with open('npMnistParameters.npy', 'wb') as f:
        for term in net:
            layer_string.append(term.name)
            if term.name == 'linear':
                layer_paramters.append((int(term.parameters[0].shape[0]), int(term.parameters[0].shape[1])))
                np.save(f, term.parameters[0])
                np.save(f, term.parameters[1])
            else:
                layer_paramters.append(None)

    with open('npMnistStructure.json', 'w') as f:
        config = {'struct': layer_string, 'num_parametes': layer_paramters}
        json.dump(config, f)

