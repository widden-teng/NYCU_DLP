import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import argparse

class Adam_class:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, w, grad_wrt_w):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_wrt_w ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w
    

class simple_CNN(object):
    def __init__(self, active_function = "sigmoid", learning_rate = 0.01, epoches = 500, optimization = "SGD"):
              
        # set the size of layer
        self.ip_size = 2
        self.h1_size = 50
        self.h2_size = 50
        self.op_size = 1

        #init the weight
        # self.w1 = np.random.randn(self.ip_size, self.h1_size)
        # self.w2 = np.random.randn(self.h1_size, self.h2_size)
        # self.w3 = np.random.randn(self.h2_size, self.op_size)
        self.w1 = np.random.normal(size = (self.ip_size, self.h1_size))
        self.w2 = np.random.normal(size = (self.h1_size, self.h2_size))
        self.w3 = np.random.normal(size = (self.h2_size, self.op_size))

        #other class variable
        self.acc = 0
        self.Loss_list = []
        self.x_train = []
        self.y_train = []
        self.datatype = ""
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.active_function = active_function
        self.optimization = optimization
        if (self.optimization == "Adam"):
            self.adam_w3 = Adam_class()
            self.adam_w2 = Adam_class()
            self.adam_w1 = Adam_class()
            self.adam_b3 = Adam_class()
            self.adam_b2 = Adam_class()
            self.adam_b1 = Adam_class()
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon=1e-8
            self.m = None
            self.v = None
            self.t = 0
        
    def init_bias(self, data_shape):
        self.b1 = np.random.randn(data_shape, self.h1_size)
        self.b2 = np.random.randn(data_shape, self.h2_size)
        self.b3 = np.random.randn(data_shape, self.op_size)

    def generate_linear(self, n=100):
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0] - pt[1])/1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)
    
    def generate_XOR_easy(self):
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)
            if 0.1*i == 0.5:
                continue
            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)
        return np.array(inputs), np.array(labels).reshape(21, 1)

    def get_linear_data(self):
        self.datatype = "linear"
        self.x_train, self.y_train = self.generate_linear(n=100)
        self.init_bias(self.x_train.shape[0])

    def get_XOR_data(self):
        self.datatype = "XOR"
        self.x_train, self.y_train = self.generate_XOR_easy()
        self.init_bias(self.x_train.shape[0])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative_sigmoid(self, x):
        return np.multiply(x, 1-x)
    
    def Relu(self, x):
        return np.maximum(0, x)

    def derivative_Relu(self, x):
        d_y = np.zeros_like(x)
        d_y[x > 0] = 1
        d_y[x == 0] = 0
        return d_y

    def forward(self):
        if self.active_function == "sigmoid" :
            self.x1 = np.dot(self.x_train, self.w1) + self.b1
            # self.x1 = np.dot(self.x_train, self.w1)
            self.a1 = self.sigmoid(self.x1)
            self.x2 = np.dot(self.a1, self.w2) + self.b2
            # self.x2 = np.dot(self.a1, self.w2)
            self.a2 = self.sigmoid(self.x2)
            self.x3 = np.dot(self.a2, self.w3) + self.b3
            # self.x3 = np.dot(self.a2, self.w3)
            self.y_pred = self.sigmoid(self.x3)
        elif self.active_function == "Relu":
            self.x1 = np.dot(self.x_train, self.w1) + self.b1
            # self.x1 = np.dot(self.x_train, self.w1)
            self.a1 = self.Relu(self.x1)
            self.x2 = np.dot(self.a1, self.w2) + self.b2
            # self.x2 = np.dot(self.a1, self.w2)
            self.a2 = self.Relu(self.x2)
            self.x3 = np.dot(self.a2, self.w3) + self.b3
            # self.x3 = np.dot(self.a2, self.w3)
            self.y_pred = self.sigmoid(self.x3)
        elif self.active_function == "None":
            self.x1 = np.dot(self.x_train, self.w1) + self.b1
            # self.x1 = np.dot(self.x_train, self.w1)
            self.x2 = np.dot(self.x1, self.w2) + self.b2
            # self.x2 = np.dot(self.x1, self.w2)
            self.x3 = np.dot(self.x2, self.w3) + self.b3
            # self.x3 = np.dot(self.x2, self.w3)
            self.y_pred = self.x3

        return self.y_pred
    
    def MSE_Loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def back_propogation(self):
        self.Loss_list.append(self.MSE_Loss(self.y_train, self.y_pred))

        if self.active_function == "sigmoid" :
            loss_grad = -2*(self.y_train - self.y_pred)
            w3_forward = loss_grad * self.derivative_sigmoid(self.y_pred)
            w3_grad = np.dot(self.a2.T, w3_forward)
            w2_forward = w3_forward.dot(self.w3.T)*self.derivative_sigmoid(self.a2)
            w2_grad =  np.dot(self.a1.T, w2_forward)
            w1_forward = np.dot(w2_forward,self.w2.T)*self.derivative_sigmoid(self.a1)
            w1_grad = np.dot(self.x_train.T,w1_forward)

        elif self.active_function == "Relu" :
            loss_grad = -2*(self.y_train - self.y_pred)
            w3_forward = loss_grad * self.derivative_sigmoid(self.y_pred)
            w3_grad = np.dot(self.a2.T, w3_forward)
            w2_forward = w3_forward.dot(self.w3.T)*self.derivative_Relu(self.a2)
            w2_grad =  np.dot(self.a1.T, w2_forward)
            w1_forward = np.dot(w2_forward,self.w2.T)*self.derivative_Relu(self.a1)
            w1_grad = np.dot(self.x_train.T,w1_forward)

        elif self.active_function == "None":
            w3_forward = loss_grad = -2*(self.y_train - self.y_pred)
            w3_grad = np.dot(self.x2.T, w3_forward)
            w2_forward = w3_forward.dot(self.w3.T)
            w2_grad =  np.dot(self.x1.T, w2_forward)
            w1_forward = np.dot(w2_forward,self.w2.T)
            w1_grad = np.dot(self.x_train.T,w1_forward)

        # #updating weight
        if (self.optimization == "Adam"):
            self.w3 = self.adam_w3.update(self.w3, w3_grad)
            self.w2 = self.adam_w2.update(self.w2, w2_grad)
            self.w1 = self.adam_w1.update(self.w1, w1_grad)
            self.b3 -= self.adam_b3.update(self.b3, w3_forward)
            self.b2 -= self.adam_b2.update(self.b2, w2_forward)
            self.b1 -= self.adam_b1.update(self.b1, w1_forward)
        elif(self.optimization == "SGD"):
            self.w3 -= self.learning_rate * w3_grad
            self.w2 -= self.learning_rate * w2_grad
            self.w1 -= self.learning_rate * w1_grad
            self.b3 -= self.learning_rate * w3_forward
            self.b2 -= self.learning_rate * w2_forward
            self.b1 -= self.learning_rate * w1_forward


        

    

    def train(self):
        for i in range(self.epoches):
            # self.get_linear_data()
            self.forward()
            self.back_propogation()
            if (i + 1) % 500 == 0:
                print("epoch {} {} loss : {}".format(i+1, self.datatype, self.Loss_list[i]))
                if i == self.epoches -1:
                    self.y_final = copy.deepcopy(self.y_pred)
                for j in range(self.x_train.shape[0]):
                    if self.y_pred[j] <= 0.5:
                        self.y_pred[j] = 0
                    else:
                        self.y_pred[j] = 1
                acc_count =len(self.y_pred[self.y_pred==self.y_train])
                # print(" Accurarcy is : {}%".format(acc_count/self.y_pred.shape[0]*100))
                self.acc = acc_count
        print("linear predict:\n", self.y_final)
        print(" Accurarcy is : {}%".format(self.acc/self.y_pred.shape[0]*100))
        plt.plot(self.Loss_list)
        plt.title("Linear Loss curve with lr={}".format(self.learning_rate))
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.show()
        self.show_result()
        # self.loss_list = []

    def show_result(self):
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for i in range(self.x_train.shape[0]):
            if self.y_train[i] == 0:
                plt.plot(self.x_train[i][0], self.x_train[i][1], 'ro')
            else:
                plt.plot(self.x_train[i][0], self.x_train[i][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(self.x_train.shape[0]):
            if self.y_pred[i] <= 0.5:
                plt.plot(self.x_train[i][0], self.x_train[i][1], 'ro')
            else:
                plt.plot(self.x_train[i][0], self.x_train[i][1], 'bo')
        plt.show()


if __name__ == "__main__":
    #select active function
    parser = argparse.ArgumentParser(description="select the active function (sigmoid, Relu, None)")
    parser.add_argument('--active_function', default = "sigmoid", help = "active function type")
    parser.add_argument('--learning_rate',type = float , default = 0.01)
    parser.add_argument('--epoches',type = int , default = 5000)
    parser.add_argument('--optimization', default = "SGD")
    

    args = parser.parse_args()

    # for linear data
    linear_demo = simple_CNN(args.active_function, args.learning_rate, args.epoches, args.optimization)
    linear_demo.get_linear_data()
    linear_demo.train()

    #for XOR data
    XOR_demo = simple_CNN(args.active_function, args.learning_rate, args.epoches, args.optimization)
    XOR_demo.get_XOR_data()
    XOR_demo.train()



    
