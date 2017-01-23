import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import xlrd

num_points = 100
fraction_to_leave_out = 0.10
input_data_range = 2*np.pi

number_of_epochs = 1000
num_iterations = 2
learning_rate = 0.1
output_col = 0


workbook = xlrd.open_workbook('TestSheet.xlsx')
sheet_names = workbook.sheet_names()
sheet = workbook.sheet_by_name(sheet_names[0])
temp_error = None



def initialize_data():
    x_data = []
    y_data = []

    data = []
    col_name = []
    row_name = []

    for col_idx in range(sheet.ncols):
        temp = []
        for row_idx in range(sheet.nrows):
            cell = sheet.cell(row_idx, col_idx)
            # print cell.value
            if (row_idx == 0):
                col_name.append(cell.value)
            else:
                temp.append(cell.value)
        if col_idx == 0:
            row_name.extend(temp)
        else:
            data.append(temp)

    num_points = row_idx - 1
    return data

def make_training_data(data):
    # Leave out random 20% of points to function as test data for fit
    #x_data, y_data = initialize_data()

    random_indices_chosen = []
    count = 0

    xTests = [[] for x in range(sheet.ncols - 2)]
    xTrains = [[] for x in range(sheet.ncols - 2)]
    yTest = []
    yTrain = []

    # Make test data set from total data set
    while count < fraction_to_leave_out*num_points:
        random_index = np.random.randint(0, num_points)
        if random_index not in random_indices_chosen:
            skipped = False
            for col in range (0, len(data)):
                if col == output_col:
                    yTest.append(data[col][random_index])
                    skipped = True
                else:
                    if (skipped == False):
                        xTests[col].append(data[col][random_index])
                    else:
                        xTests[col-1].append(data[col][random_index])
            count += 1
            random_indices_chosen.append(random_index)
        else:
            random_index = np.random.randint(0, num_points)

    # Make training data set from those points not chosen to be in test data set
    for index in range(num_points):
        if index not in random_indices_chosen:
            skipped = True
            for col in range (0, len(data)):
                if col == output_col:
                    yTrain.append(data[col][index])
                    skipped = True
                else:
                    if (skipped == False):
                        xTrains[col].append(data[col][index])
                    else:
                        xTrains[col-1].append(data[col][index])
    return xTrains, yTrain, xTests, yTest

def make_array_from_list(xTrains, yTrain, xTests, yTest):

    xTrains_array = []
    xTests_array =[]
    for train in xTrains:
        xTrains_array.append(np.asarray(train).reshape(1,-1))

    for test in xTests:
        xTests_array.append(np.asarray(test).reshape(1,-1))


    xTrain_array = xTrains_array[0]
    yTrain_array = np.asarray(yTrain).reshape([1, -1])
    xTest_array = xTests_array[0]
    yTest_array = np.asarray(yTest).reshape([1, -1])

    return xTrains_array, yTrain_array, xTests_array, yTest_array

def launch_tensorflow(xTrains, yTrain, xTests, yTest, data):
    # Setup Tensorflow session

    # Layer 1
    placeholders = []
    for i in range(0, len(xTrains)):
        placeholders.append(tf.placeholder(tf.float32, [1, None]))

    W = tf.Variable(tf.truncated_normal(shape=[10, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10, 1]))

    # Layer 2 (Hidden layer)
    W2 = tf.Variable(tf.truncated_normal(shape=[1, 10], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    # Activation (layer 1 -> layer 2)
    temp = b
    for p in range(0, len(placeholders)):
        temp += tf.matmul(W, placeholders[p])
    hidden_layer = tf.nn.sigmoid(temp)

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

        feed_dict = {}
        for index in range(0, len(xTrains)):
            feed_dict[placeholders[index]] = xTrains[index]
        sess.run(optimizer, feed_dict=feed_dict)
        if epoch % 100 == 0:
            print ("Epoch number", epoch, "Training set RMSE:", cost.eval(feed_dict, sess))



    yFit_feed = {}
    test_yFit_feed = {}
    train_yFit_feed = {}
    for i in range (0, len(placeholders)):
        yFit_feed[placeholders[i]] = np.asarray(data[i]).reshape([1, -1])
        test_yFit_feed[placeholders[i]] = np.asarray(xTests[i]).reshape([1, -1])
        train_yFit_feed[placeholders[i]] = np.asarray(xTrains[i]).reshape([1, -1])

    yFit =       y.eval(yFit_feed, sess)
    test_yFit =  y.eval(test_yFit_feed, sess)
    train_yFit = y.eval(train_yFit_feed, sess)

    residual_test = test_yFit - yTest
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))
    print ("Testing set RMSE:", rmse_test)

    return yFit, rmse_test, test_yFit, train_yFit

def plot_data(label, xTrains_list, yTrain_list, xTests_list, yTest_list, yFit, data, test_yFit, train_yFit):

    yData = data[output_col]

    x = tf.placeholder(tf.float32, [1, None])
    yFit_list = yFit.transpose().tolist()
    test_yFit_list =  np.asarray(test_yFit.transpose().tolist()).flatten().tolist()
    train_yFit_list = np.asarray(train_yFit.transpose().tolist()).flatten().tolist()

    residual_test = test_yFit -yTest_list
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))


    fig1 = plt.figure()

    #ACTUAL VS PREDICTED
    ax2 = fig1.add_subplot(221)
    plt.plot(train_yFit_list, yTrain_list, 'ro', label='training data')
    plt.plot(test_yFit_list,  yTest_list, 'bo', label='test data')
    ax2.set_xlabel('PredictedY')
    ax2.set_ylabel('ActualY')
    ax2.set_title(label + ' ActualY vs PredictedY')
    textstr1='Testing\n$RMSE=%.10f$'%(rmse_test)
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax2.legend()
    plt.draw()

    #HISTOGRAM
    yFit_list = np.asarray(yFit_list).flatten().tolist()
    ax4 = fig1.add_subplot(224)
    n, bins, patches = plt.hist([yData, yFit_list], 10, normed=1, histtype='bar', color=['orange', 'red'],label=['Actual','Predicted'])
    textstr1='Predicted\n$Min=%.10f$\n$Max=%.10f$\n$Mean=%.10f$\n$Std Dev=%.10f$'%(min(yFit_list),max(yFit_list),np.mean(yFit_list),np.std(yFit_list))
    textstr2='Actual\n$Min=%.10f$\n$Max=%.10f$\n$Mean=%.10f$\n$Std Dev=%.10f$'%(min(yData),max(yData),np.mean(yData),np.std(yData))
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax4.text(0.05,0.95,textstr1,transform=ax4.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax4.text(0.05,0.60,textstr2,transform=ax4.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax4.legend()
    ax4.set_title(label + ' Predicted Compared with Actual Histogram')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Frequency')
    plt.legend()


def main():

    data = initialize_data()

    #TODO
    xTrains, yTrain, xTests, yTest = make_training_data(data=data)

    xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)


    min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains, yTrain, xTests, yTest
    min_xTrains_array, min_yTrain_array, min_xTests_array, min_yTest_array = xTrains_array, yTrain_array, xTests_array, yTest_array

    min_error = float("inf")
    min_yFit = None
    min_test_yFit = None
    min_train_yFit = None

    max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains, yTrain, xTests, yTest
    max_xTrains_array, max_yTrain_array, max_xTests_array, max_yTest_array = xTrains_array, yTrain_array, xTests_array, yTest_array

    max_error = float("-inf")
    max_yFit = None
    max_test_yFit = None
    max_train_yFit = None

    rmses = []

    for x in range(0,num_iterations):
        yFit, rmse_test, test_yFit, train_yFit = launch_tensorflow(xTrains=xTrains_array, yTrain=yTrain_array, xTests=xTests_array, yTest=yTest_array, data=data)
        rmses.append(rmse_test)
        if rmse_test < min_error:
        	min_error = rmse_test
        	min_yFit = yFit.copy()
        	min_test_yFit = test_yFit.copy()
        	min_train_yFit = train_yFit.copy()
        	min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains, yTrain, xTests, yTest

        if rmse_test > max_error:
        	max_error = rmse_test
        	max_yFit = yFit.copy()
        	max_test_yFit = test_yFit.copy()
        	max_train_yFit = train_yFit.copy()
        	max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains, yTrain, xTests, yTest


        #TODO
        xTrains, yTrain, xTests, yTest = make_training_data(data=data)

        xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)


    for x in range(0,len(rmses)):
    	 'Iteration ', x, ': ', rmses[x]
    print ('Best RMSE: ', min_error)
    print ('Worst RMSE: ', max_error)
    print ("Mean RMSE: ", np.array(rmses).mean())
    print ("Std RMSE: ", np.array(rmses).std())
    plot_data(label='Best', xTrains_list=min_xTrains, yTrain_list=min_yTrain, xTests_list=min_xTests, yTest_list=min_yTest, yFit=min_yFit, data=data, test_yFit = min_test_yFit, train_yFit = min_train_yFit)
    plot_data(label='Worst', xTrains_list=max_xTrains, yTrain_list=max_yTrain, xTests_list=max_xTests, yTest_list=max_yTest, yFit=max_yFit, data=data, test_yFit = max_test_yFit, train_yFit = max_train_yFit)

    plt.show()
main()
