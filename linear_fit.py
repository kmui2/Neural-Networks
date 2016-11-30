import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
# import plotly.plotly as py  # tools to communicate with Plotly's server

# Train a data set
size_data_array = [ 2104,  1600,  2400,  1416,  3000,  1985,  1534,  1427,
  1380,  1494,  1940,  2000,  1890,  4478,  1268,  2300,
  1320,  1236,  2609,  3031,  1767,  1888,  1604,  1962,
  3890,  1100,  1458,  2526,  2200,  2637,  1839,  1000,
  2040,  3137,  1811,  1437,  1239,  2132,  4215,  2162,
  1664,  2238,  2567,  1200,   852,  1852,  1203 ]
size_data = numpy.asarray(size_data_array)
price_data_array = [ 399900,  329900,  369000,  232000,  539900,  299900,  314900,  198999,
  212000,  242500,  239999,  347000,  329999,  699900,  259900,  449900,
  299900,  199900,  499998,  599000,  252900,  255000,  242900,  259900,
  573900,  249900,  464500,  469000,  475000,  299900,  349900,  169900,
  314900,  579900,  285900,  249900,  229900,  345000,  549000,  287000,
  368500,  329900,  314000,  299000,  179900,  299900,  239500 ]
price_data = numpy.asarray(price_data_array)
size_data_test = numpy.array([])
price_data_test = numpy.array([])
test_num = 6
# Test a data set
for x in xrange(0,test_num):
	rnd = numpy.random.randint(0,47)
	size_data_test = numpy.append(size_data_test,size_data_array[rnd])
	price_data_test = numpy.append(price_data_test,price_data_array[rnd])

def normalize(array): 
    return (array - array.mean()) / array.std()

# Normalize a data set

size_data_n = normalize(size_data)
price_data_n = normalize(price_data)

size_data_test_n = normalize(size_data_test)
price_data_test_n = normalize(price_data_test)


samples_number = price_data_n.size

# TF graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create a model

# Set model weights
W = tf.Variable(numpy.random.randn(), name="weight")
b = tf.Variable(numpy.random.randn(), name="bias")

# Set parameters
learning_rate = 0.1
training_iteration = 200

# Construct a linear model
model = tf.add(tf.mul(X, W), b)

# Minimize squared errors
cost_function = tf.reduce_sum(tf.pow(model - Y, 2))/(2 * samples_number) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) #Gradient descent

# Initialize variables
init = tf.initialize_all_variables()

mean = numpy.mean(price_data)
std = numpy.std(price_data)

def rms(predictions, targets):
	sum = 0
	for x in xrange(0,len(predictions)):
		sum += (predictions[x] - targets[x])**2
	mean = sum/len(predictions)
	return numpy.sqrt(mean)

# # Polynomial Regression
# def polyfit(x, y, degree):
#     results = {}

#     coeffs = numpy.polyfit(x, y, degree)

#      # Polynomial Coefficients
#     results['polynomial'] = coeffs.tolist()

#     # r-squared
#     p = numpy.poly1d(coeffs)
#     # fit values, and mean
#     yhat = p(x)                         # or [p(z) for z in x]
#     ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
#     ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
#     sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
#     results['determination'] = ssreg / sstot

#     return results

# Launch a graph
temp_size = [0]*len(size_data_n)
temp_price = [0]*len(price_data_n)
save_size = numpy.copy(size_data_n)
save_price = numpy.copy(price_data_n)
num_sets = 2
rms_array = [0]*num_sets
with tf.Session() as sess:
    for a in xrange(0,num_sets):
	    sess.run(init)
	    display_step = 20
	    # Fit all training data
	    for iteration in range(training_iteration):
	        for (x, y) in zip(size_data_n, price_data_n):
	            sess.run(optimizer, feed_dict={X: x, Y: y})

	        # Display logs per iteration step
	        if iteration % display_step == 0:
	            for z in xrange(0,len(size_data_n)):
	            	temp_price[z]=(size_data_n[z]*sess.run(W) + sess.run(b))*std+mean
	            print "Iteration:", '%04d' % (iteration + 1), "cost=", rms(temp_price,price_data),\
	            "W=", sess.run(W)*std, "b=", sess.run(b)*std+mean
	            
	    tuning_cost = sess.run(cost_function, feed_dict={X: normalize(size_data_n), Y: normalize(price_data_n)})
	            
	    print "Tuning completed:", rms(temp_price,price_data), "W=", sess.run(W)*std, "b=", sess.run(b)*std+mean
	    
	    # Validate a tuning model
	    

	    predicted_test = [0]*len(size_data_test_n)
	    for x in xrange(0,len(size_data_test_n)):
	    	predicted_test[x]=(size_data_test_n[x]*sess.run(W)+sess.run(b))*std+mean
	    testing_cost = rms(predicted_test,price_data_test)
	    rms_array[a]=[testing_cost,sess.run(W),sess.run(b)]

	    print "Testing data cost:" , testing_cost
	    # size_data_n=numpy.copy(save_size)
	    # price_data_n=numpy.copy(save_price)

	    for x in xrange(0,test_num):
	    	rnd = numpy.random.randint(0,47)
	    	size_data_test = numpy.append(size_data_test,size_data_array[rnd])
	    	price_data_test = numpy.append(price_data_test,price_data_array[rnd]) 
	    size_data_test_n = normalize(size_data_test)
	    price_data_test_n = normalize(price_data_test)


    temp_min = float("inf")
    temp_max = float("-inf")
    rms_all = [0]*len(rms_array)
    set_num = -1
    for x in xrange(0,len(rms_array)):
    	if rms_array[x][0] < temp_min:
    		temp_min=rms_array[x][0]
    		set_num=x
    	if rms_array[x][0] > temp_max:
    		temp_max=rms_array[x][0]
		rms_all[x] = rms_array[x][0]
    print 'Best RMS = ', temp_min
    print 'Worst RMS = ', temp_max
    print 'avg RMS = ', numpy.mean(rms_all)
    print 'std RMS = ', numpy.std(rms_all)
    W_num =  rms_array[set_num][1]
    b_num = rms_array[set_num][2]
    mean = numpy.mean(price_data)
    std = numpy.std(price_data)
    predicted_data_array = [0] * len(size_data_array)
    for x in range(0,len(size_data_array)):
        predicted_data_array[x] = (W_num * size_data_n[x] + b_num)*std+mean;

    # # Display a plot
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.set_title('Normalized Data')
    # ax2.set_xlabel('xMean-xData')
    # ax2.set_ylabel('yMean-yData')
    # plt.plot(size_data_n, price_data_n, 'ro', label='Normalized samples')
    # plt.plot(size_data_test_n, price_data_test_n, 'go', label='Normalized testing samples')
    # plt.plot(size_data_n, W_num * size_data_n + b_num, label='Fitted line')
    # plt.legend()
    # plt.draw()


    # Display a plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Raw Data')
    ax1.set_xlabel('xData')
    plt.plot(size_data, price_data, 'ro', label='Samples data')
    plt.plot(size_data, numpy.asarray(predicted_data_array), label='Fitted Line')
    textstr='$y=%.2fx+%.2f$\n$RMS=%.2f$'%(W_num*std,b_num*std+mean,rms(predicted_data_array,price_data_array))
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax1.text(0.05,0.95,textstr,transform=ax1.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    plt.legend()
    plt.draw()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title('Predicted vs. Actual Y')
    ax3.set_xlabel('Actual Y')
    ax3.set_ylabel('Predicted')
    plt.plot(price_data, numpy.asarray(predicted_data_array),'ro', label='Predicted')
    ax3.plot(ax3.get_xlim(), ax3.get_ylim(), ls="--", c=".3")
    plt.legend()
    plt.draw()

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_title('Predicted Diff vs. xData')
    ax4.set_xlabel('xData')
    ax4.set_ylabel('Predicted Diff')
    predicted_data_diff_array = [0] * len(size_data_array)
    for x in range(0,len(size_data_array)):
        predicted_data_diff_array[x] = predicted_data_array[x]-price_data_array[x];
    plt.plot(size_data, numpy.asarray(predicted_data_diff_array),'ro', label='Predicted')
    plt.axhline(0, color='black')
    plt.legend()
    plt.draw()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = plt.hist([price_data_array, predicted_data_array], 10, normed=1, histtype='bar', 
    	color=['orange', 'red'],label=['Actual','Predicted'])
    textstr1='Predicted\n$Min=%.2f$\n$Max=%.2f$\n$Mean=%.2f$\n$Std Dev=%.2f$'%(min(predicted_data_array),max(predicted_data_array),numpy.mean(predicted_data_array),numpy.std(predicted_data_array))
    textstr2='Actual\n$Min=%.2f$\n$Max=%.2f$\n$Mean=%.2f$\n$Std Dev=%.2f$'%(min(price_data_array),max(price_data_array),mean,std)
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax.text(0.05,0.95,textstr1,transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax.text(0.05,0.60,textstr2,transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    plt.legend()
    ax.set_title('Predicted Compared with Actual Histogram')
    ax.set_xlabel('Y')
    ax.set_ylabel('Frequency')
    plt.legend()
    plt.show()
