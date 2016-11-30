import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

num_points = 100
fraction_to_leave_out = 0.2
input_data_range = 2*np.pi

number_of_epochs = 1000
num_iterations = 10
learning_rate = 0.1


temp_error = None

def initialize_data():
    x_data = []
    y_data = []

    x_data = np.linspace(0.0, input_data_range, num_points)
    y_data = np.sin(x_data)

    #print x_data
    return x_data, y_data

def make_training_data(x_data, y_data):
    # Leave out random 20% of points to function as test data for fit

    #x_data, y_data = initialize_data()

    xTest = []
    yTest = []
    xTrain = []
    yTrain = []
    random_indices_chosen = []
    count = 0

    # Make test data set from total data set
    while count < fraction_to_leave_out*num_points:
        random_index = np.random.randint(0, num_points)
        if random_index not in random_indices_chosen:
            #print "The random index is", random_index
            xTest.append(x_data[random_index])
            yTest.append(y_data[random_index])
            count += 1
            random_indices_chosen.append(random_index)
        else:
            random_index = np.random.randint(0, num_points)

    # Make training data set from those points not chosen to be in test data set
    for index in range(num_points):
        if index not in random_indices_chosen:
            xTrain.append(x_data[index])
            yTrain.append(y_data[index])
    return xTrain, yTrain, xTest, yTest

def make_array_from_list(xTrain, yTrain, xTest, yTest):
    xTrain_array = np.asarray(xTrain).reshape([1, -1])
    yTrain_array = np.asarray(yTrain).reshape([1, -1])
    xTest_array = np.asarray(xTest).reshape([1, -1])
    yTest_array = np.asarray(yTest).reshape([1, -1])

    return xTrain_array, yTrain_array, xTest_array, yTest_array

def launch_tensorflow(xTrain, yTrain, xTest, yTest, xData):
    # Setup Tensorflow session

    # Layer 1
    x = tf.placeholder(tf.float32, [1, None])
    W = tf.Variable(tf.truncated_normal(shape=[10, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10, 1]))

    # Layer 2 (Hidden layer)
    W2 = tf.Variable(tf.truncated_normal(shape=[1, 10], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    # Activation (layer 1 -> layer 2)
    hidden_layer = tf.nn.sigmoid(tf.matmul(W, x) + b)

    # Output from layer 2 (hidden layer)
    y = tf.matmul(W2, hidden_layer) + b2

    # Minimize the squared errors.
    cost = tf.reduce_mean(tf.square(y - yTrain))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    # Launch TensorFlow graph
    sess = tf.Session()
    # with tf.Session() as sess:
    sess.run(init)
    for epoch in range(number_of_epochs + 1):
        sess.run(optimizer, {x: xTrain})
        # optimizer.run({x: xTrain}, sess)
        if epoch % 100 == 0:
            print "Epoch number", epoch, "Training set RMSE:", cost.eval({x: xTrain}, sess)
    
    xData = np.asarray(xData).reshape([1, -1])
    test_yFit =  y.eval({x: xTest}, sess)
    train_yFit = y.eval({x: xTrain}, sess)
    yFit =       y.eval({x: xData}, sess)

    residual_test = test_yFit - np.sin(xTest)
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))
    print "Testing set RMSE:", rmse_test

    return [yFit, rmse_test, test_yFit, train_yFit]

def plot_data(label, xTrain_list, yTrain_list, xTest_list, yTest_list, yFit, xData, yData, test_yFit, train_yFit):

    x = tf.placeholder(tf.float32, [1, None])
    yFit_list = yFit.transpose().tolist()
    test_yFit_list =  np.asarray(test_yFit.transpose().tolist()).flatten().tolist()
    train_yFit_list = np.asarray(train_yFit.transpose().tolist()).flatten().tolist()


    #RAW DATA
    fig1 = plt.figure()
    plt.plot(xData, yFit_list, 'b-', label='model fit')
    plt.plot(xTrain_list, yTrain_list, 'ro', label='training data')
    plt.plot(xTest_list, yTest_list, 'bo', label='test data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(label + ' Training and Testing Data')
    plt.legend()
    plt.draw()

    #ACTUAL VS PREDICTED
    fig3 = plt.figure()
    plt.plot(train_yFit_list, yTrain_list, 'ro', label='training data')
    plt.plot(test_yFit_list,  yTest_list, 'bo', label='test data')
    plt.xlabel('PredictedY')
    plt.ylabel('ActualY')
    plt.title(label + ' ActualY vs PredictedY')
    plt.legend()
    plt.draw()

    #RESIDUALS
    fig2 = plt.figure()
    test_residuals = []
    for x in xrange(0,len(test_yFit_list)):
    	test_residuals.append(yTest_list[x]-test_yFit_list[x])
    train_residuals = []
    for x in xrange(0,len(train_yFit_list)):
    	train_residuals.append(yTrain_list[x]-train_yFit_list[x])
    plt.plot(xTrain_list, train_residuals, 'ro', label='training data')
    plt.plot(xTest_list, test_residuals, 'bo', label='test data')
    plt.xlabel('x')
    plt.ylabel('yTest - yPredicted')
    plt.title(label + ' Residuals')
    plt.legend()
    plt.draw()

    #HISTOGRAM
    yFit_list = np.asarray(yFit_list).flatten().tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = plt.hist([yData, yFit_list], 10, normed=1, histtype='bar', color=['orange', 'red'],label=['Actual','Predicted'])
    textstr1='Predicted\n$Min=%.2f$\n$Max=%.2f$\n$Mean=%.2f$\n$Std Dev=%.2f$'%(min(yFit_list),max(yFit_list),np.mean(yFit_list),np.std(yFit_list))
    textstr2='Actual\n$Min=%.2f$\n$Max=%.2f$\n$Mean=%.2f$\n$Std Dev=%.2f$'%(min(yData),max(yData),np.mean(yData),np.std(yData))
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax.text(0.05,0.95,textstr1,transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax.text(0.05,0.60,textstr2,transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    plt.legend()
    ax.set_title(label + ' Predicted Compared with Actual Histogram')
    ax.set_xlabel('Y')
    ax.set_ylabel('Frequency')
    plt.legend()


def main():

    x_data, y_data = initialize_data()
    xTrain, yTrain, xTest, yTest = make_training_data(x_data=x_data, y_data=y_data)
    xTrain_array, yTrain_array, xTest_array, yTest_array = make_array_from_list(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)
    
    min_xTrain, min_yTrain, min_xTest, min_yTest = xTrain, yTrain, xTest, yTest
    min_xTrain_array, min_yTrain_array, min_xTest_array, min_yTest_array = xTrain_array, yTrain_array, xTest_array, yTest_array

    min_error = float("inf")
    min_yFit = None
    min_test_yFit = None
    min_train_yFit = None

    max_xTrain, max_yTrain, max_xTest, max_yTest = xTrain, yTrain, xTest, yTest
    max_xTrain_array, max_yTrain_array, max_xTest_array, max_yTest_array = xTrain_array, yTrain_array, xTest_array, yTest_array

    max_error = float("-inf")
    max_yFit = None
    max_test_yFit = None
    max_train_yFit = None

    rmses = []

    for x in xrange(0,num_iterations):
        temp = launch_tensorflow(xTrain=xTrain_array, yTrain=yTrain_array, xTest=xTest_array, yTest=yTest_array, xData=x_data)
        rmses.append(temp[1])
        if temp[1] < min_error:
        	min_error = temp[1]
        	min_yFit = temp[0].copy()
        	min_test_yFit = temp[2].copy()
        	min_train_yFit = temp[3].copy()
        	min_xTrain, min_yTrain, min_xTest, min_yTest = list(xTrain), list(yTrain), list(xTest), list(yTest)

        if temp[1] > max_error:
        	max_error = temp[1]
        	max_yFit = temp[0].copy()
        	max_test_yFit = temp[2].copy()
        	max_train_yFit = temp[3].copy()
        	max_xTrain, max_yTrain, max_xTest, max_yTest = list(xTrain), list(yTrain), list(xTest), list(yTest)
        	

        xTrain, yTrain, xTest, yTest = make_training_data(x_data=x_data, y_data=y_data)
        xTrain_array, yTrain_array, xTest_array, yTest_array = make_array_from_list(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)
    
    print "Mean RMSE: ", np.array(rmses).mean()
    print "Std RMSE: ", np.array(rmses).std()
    plot_data(label='best', xTrain_list=min_xTrain, yTrain_list=min_yTrain, xTest_list=min_xTest, yTest_list=min_yTest, yFit=min_yFit, xData = x_data, yData = y_data, test_yFit = min_test_yFit, train_yFit = min_train_yFit)
    plot_data(label='worst', xTrain_list=max_xTrain, yTrain_list=max_yTrain, xTest_list=max_xTest, yTest_list=max_yTest, yFit=max_yFit, xData = x_data, yData = y_data, test_yFit = max_test_yFit, train_yFit = max_train_yFit)

    plt.show()
main()