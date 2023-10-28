#手搓赛博算命器
import numpy as np
import ANNnet2 as cp
import copy
import math

NETWORK_SHAPE = [2, 100, 200, 500, 2]
BATCH_SIZE = 30
LEARNING_RATE = 0.015  #学习率
NUM_OF_DATA = 800
force_train = False
random_train = False


n_improved = 0
n_not_improved = 0

#标准化函数
def normalize(array):
    max_number = np.max(np.absolute(array),axis=1,keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm

#向量标准化函数
def vector_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm
#激活函数
def activation_ReLU(inputs):
    return np.maximum(0,inputs)
#分类函数
def classify(probabilities):
    classification = np.rint(probabilities[:,1])
    return classification

#softmax激活函数
def activation_softmax(inputs):
    max_values = np.max(inputs,axis=1,keepdims=True)
    #将数据滑动到负值（防止指数爆炸）
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    #标准化#axis使得可以纵向求和
    norm_base = np.sum(exp_values,axis=1,keepdims=True)
    norm_values = exp_values/norm_base
    return norm_values

#损失函数1
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real),2))
    real_matrix[:,1] = real
    real_matrix[:,0] = 1-real
    product = 1-np.sum(predicted*real_matrix, axis = 1)
    return product
#损失函数2
def loss_function(predicted, real):
    condition = (predicted>0.5)
    binary_predicted = np.where(condition, 1, 0)
    real_matrix = np.zeros((len(real),2))
    real_matrix[:,1] = real
    real_matrix[:,0] = 1-real
    product = 1-np.sum(predicted*real_matrix, axis = 1)
    return product

#需求函数
def get_final_layer_preAct_demands(predicted_values, target_vector):
    target = np.zeros((len(target_vector),2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector

    for i in range(len(target_vector)):
        if np.dot(target[i],predicted_values[i]) > 0.5 :
            target[i] = np.array([0, 0])
        else :
            target[i] = (target[i]-0.5)*2
    return target


#定义一个层类
class Layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def layer_backward(self, preWeight_values, afterWeights_demands):
        preWeights_demands = np.dot(afterWeights_demands, self.weights.T)

        condition = (preWeight_values > 0)
        value_derivatives = np.where(condition, 1, 0)
        preActs_demands = value_derivatives * preWeights_demands
        norm_preAct_demands = normalize(preActs_demands)

        get_weight_adjust_matrix = self.get_weight_adjust_matrix(preWeight_values, afterWeights_demands)
        norm_get_weight_adjust_matrix = normalize(get_weight_adjust_matrix)

        return (norm_preAct_demands, norm_get_weight_adjust_matrix)

    def get_weight_adjust_matrix(self, preWeights_values, afterWeights_demands):
        plain_weight = np.full(self.weights.shape, 1)
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)
        plain_weight_T = plain_weight.T

        for i in range(BATCH_SIZE):
            weights_adjust_matrix +=(plain_weight_T*preWeights_values[i, :]).T*afterWeights_demands[i, :]
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix

#定义一个网络类
class Network:
    def __init__(self,network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layer = Layer(network_shape[i],network_shape[i+1])
            self.layers.append(layer)
    #前馈运算函数
    def network_forward(self,inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i<len(self.layers)-1 :
                layer_output = activation_ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else :
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
        return outputs
    #反向传播函数
    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self)  # 备用网络
        preAct_demands = get_final_layer_preAct_demands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - (1+i)] #倒序
            if i != 0 :
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis = 0)
                layer.biases = vector_normalize(layer.biases)

            outputs = layer_outputs[len(layer_outputs) - (2+i)]
            results_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands = results_list[0]
            weights_adjust_matrix = results_list[1]
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            layer.weights = normalize(layer.weights)

        return backup_network
    #单批次训练
    def one_batch_train(self, batch):
        global force_train, random_train, n_improved, n_not_improved
        inputs = batch[:,(0,1)]
        targets = copy.deepcopy(batch[:, 2]).astype(int) #标准答案
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], targets)
        loss = loss_function(outputs[-1], targets)

        if np.mean(precise_loss) <= 0.1:
            print("no need for training")
        else:
            backup_network = self.network_backward(outputs,targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], targets)
            backup_loss = loss_function(backup_outputs[-1], targets)

            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print('Improved')
                n_improved +=1
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print("Force train")
                if random_train:
                    self.random_update()
                    print("Random update")

                print('No improvement')
                n_not_improved +=1
        print('-------------------------------------------------------')
    #多批次训练
    def train(self, n_entries):
        global force_train, random_train, n_improved, n_not_improved
        n_improved = 0
        n_not_improved = 0

        n_batches = math.ceil(n_entries/BATCH_SIZE)
        for i in range(n_batches):
            batch =cp.create_data(BATCH_SIZE)
            self.one_batch_train(batch)
        improvement_rate = n_improved/(n_not_improved+n_improved)
        print("Improvement rate:",format(improvement_rate, ".0%"))
        if improvement_rate < 0.05:
            force_train=True
        else:
            force_train = False
        if n_improved == 0:
            random_train = True
        else:
            random_train = False

        data = cp.create_data(NUM_OF_DATA)
        inputs = data[:,(0,1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:,2] = classification
        cp.plot_data(data, "After training")

    #随机更新
    def random_update(self):
        random_network = Network(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weights_change
            self.layers[i].biases += biases_change
#-----------------------------------------------

def main():
    data = cp.create_data(NUM_OF_DATA)#生成数据
    cp.plot_data(data, "Right classification")

    #选择起始网络
    use_this_network = 'n' #No
    while use_this_network != 'Y' and use_this_network != 'y':
        network = Network(NETWORK_SHAPE)
        inputs = data[:,(0,1)]
        outputs = network.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:,2] = classification
        cp.plot_data(data, "Choose network")
        use_this_network = input("Use this network! Y to yes , N to no \n")

    #进行训练
    do_train = input("Train!, Y to yes , N to no \n")
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() is True:
        if do_train.isnumeric() is True:
            n_entries = int(do_train)
        else:
            n_entries = int(input("Enter the number of data entries used to train. \n"))

        network.train(n_entries)
        do_train = input("Train!, Y to yes , N to no \n")

    #演示训练效果
    inputs = data[:, (0, 1)]
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    cp.plot_data(data, "After training")

#-----------------TEST--------------------------

#-----------------RUN---------------------------
main()