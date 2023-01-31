import numpy as np, scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, clim, semilogx, loglog, title, subplot, grid
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn import model_selection, tree
from scipy import stats
import torch
from tabulate import tabulate
from toolbox_02450 import feature_selector_lr, bmplot, rlr_validate, train_neural_net, visualize_decision_boundary
from sklearn import preprocessing
from toolbox_02450 import mcnemar

pd.set_option("display.max_rows", None, "display.max_columns", None)

Headers = ["RI" , "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]

attributeNames = ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

X = pd.read_csv("/Users/albert/Desktop/Glass.txt" , sep = "," , names = Headers)

classNames = ["building_windows_1", "building_windows_2", "vehicle_windows_1", "vehicle_windows_2", "containers", "tableware", "headlamps"]

X = X.to_numpy()

N, M = X.shape

#X = (X - np.ones((N,1))*X.mean(axis=0))/(np.ones((N,1))*X.std(axis = 0))

X = stats.zscore(X)


#We want to predict the refractive index: 

y = X[:,0]

#We want to remove the type column and the refractive index column: 

X = X[:,1:9]

C = len(classNames)

N, M = X.shape


"""
Regression part A)
"""


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-6,10))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))


"""
Regression part B)
"""

Headers = ["RI" , "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]

attributeNames = ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

X = pd.read_csv("/Users/albert/Desktop/Glass.txt" , sep = "," , names = Headers)

classNames = ["building_windows_1", "building_windows_2", "vehicle_windows_1", "vehicle_windows_2", "containers", "tableware", "headlamps"]

X = X.to_numpy()

N, M = X.shape

X = stats.zscore(X)

#We want to predict the refractive index: 
y = X[:,0]

#We want to remove the type column and the refractive index column: 

X = X[:,1:9]

C = len(classNames)

N, M = X.shape

# Create crossvalidation partition for evaluation
K1 = 10
K2 = 10
CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
baseline_average_error = np.empty((K1, 1))
baseline_test_error = np.zeros(X.shape[0])
linear_regression_average_error = np.empty((K1, 1))
linear_regression_test_error = np.zeros(X.shape[0])
ann_average_error = np.empty((K1, 1))
ann_test_error = np.zeros(X.shape[0])
mu = np.empty((K1, M))
sigma = np.empty((K1, M))
w_rlr = np.empty((M+1, K1))
opt_lambdas_rlr = np.empty((K1, 1))

n_replicates = 1 # number of networks trained in each k-fold
n_hidden_units = [1,2,3,4,5,6,7,8,9,10] # number of hidden units in the single hidden layer
max_iter = 10000 # Train for a maximum of 10000 steps, or until convergence (see help for the
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
opt_hidden_units = np.empty((K1, 1))

k = 0
for train_index, test_index in CV_outer.split(X):

    # extract training and test set for current CV fold (outer fold)
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    

    #Baseline
    baseline_prediction = y_train.mean()
    
    baseline_average_error[k] = np.square(y_test - baseline_prediction).sum(axis=0) / y_test.shape[0]
    baseline_test_error[test_index] = np.square(y_test - baseline_prediction)


    #Linear Regression 
    # Add offset attributes
    X_train_reg = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test_reg = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)

    # 10-fold cross validate linear regression to find optimal lambda (inner loop)
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_reg,
                                                                                                      y_train,
                                                                                                      lambdas,
                                                                                                      K2)
    opt_lambdas_rlr[k] = opt_lambda

    mu[k, :] = np.mean(X_train_reg[:, 1:], 0)
    sigma[k, :] = np.std(X_train_reg[:, 1:], 0)

    X_train_reg[:, 1:] = (X_train_reg[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test_reg[:, 1:] = (X_test_reg[:, 1:] - mu[k, :]) / sigma[k, :]

    # Calculate weights for the optimal value of lambda, on entire training set
    Xty = X_train_reg.T @ y_train
    XtX = X_train_reg.T @ X_train_reg

    lambdaI = opt_lambda * np.eye(M+1)
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Compute mean squared error with regularization with optimal lambda
    linear_regression_average_error[k] = np.square(y_test - X_test_reg @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    linear_regression_test_error[test_index] = np.square(y_test - X_test_reg @ w_rlr[:, k])


    #ANN

    errors = np.zeros((len(n_hidden_units), K2))  # make a list for storing generalizaition error for each model

    i = 0
    for train_index_ann, test_index_ann in CV_inner.split(X_train):

        X_train_ann = torch.Tensor(X_train[train_index_ann, :])
        y_train_ann = torch.Tensor(y_train[train_index_ann])
        X_test_ann = torch.Tensor(X_train[test_index_ann, :])
        y_test_ann = torch.Tensor(y_train[test_index_ann])

        # Model loop
        j = 0
        for n in n_hidden_units:

            inner_model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function, //todo kan ikke fjerne dette .. ?
                torch.nn.Linear(n, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(inner_model,
                                                               loss_fn,
                                                               X=X_train_ann,
                                                               y=y_train_ann,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)

            # Determine estimated class labels for test set
            y_test_est_ann = net(X_test_ann)

            # Determine errors
            se = (y_test_est_ann.float() - y_test_ann.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test_ann)).data.numpy().mean()  # mean
            errors[j][i] = mse

            j += 1
        i += 1

    opt_hidden_units[k] = n_hidden_units[np.argmin(np.mean(errors, axis=1))]
    opt_hidden_unit = opt_hidden_units[k][0].astype(int)

    # Compute error for best performing model for the outer loop
    outer_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_hidden_unit),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function, //todo kan ikke fjerne dette .. ?
        torch.nn.Linear(opt_hidden_unit, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(outer_model,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train),
                                                       y=torch.Tensor(y_train),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)

    # Determine estimated class labels for test set
    y_test_est = net(torch.Tensor(X_test))

    # Determine errors
    se = (y_test_est.float() - torch.Tensor(y_test).float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(torch.Tensor(y_test))).data.numpy().mean()  # mean
    ann_average_error[k] = mse
    ann_test_error[test_index] = np.square(y_test_est.float().data.numpy()[:,0] - torch.Tensor(y_test).float().data.numpy())

    k += 1
    
output_data = np.hstack((
    np.arange(K1).reshape(K1,1) + 1,
    opt_hidden_units,
    ann_average_error,
    opt_lambdas_rlr,
    linear_regression_average_error,
    baseline_average_error
))

print(tabulate(
    output_data,
    headers=['i','ann_h','ann_err','lr_lambda','lr_err','base_err']))

#i    ann_h    ann_err    lr_lambda     lr_err    base_err
#---  -------  ---------  -----------  ---------  ----------
#1        1   0.343173        0.01   0.044818     0.342971
#2        7   0.964132       10      0.290907     0.959367
#3        1   1.76964         0.01   0.15492      1.7678
#4        2   0.622019        0.1    0.144821     0.622085
#5        5   0.503841        1e-05  0.0642138    0.50911
#6        4   1.15459         0.1    0.0838095    1.15266
#7        3   0.991147        0.01   0.100107     0.988584
#8        6   0.692047        0.1    0.118358     0.690123
#9        2   0.389072        0.01   0.102987     0.396148
#10        2   2.68452         0.1    0.227423     2.68535
#(array([0.38675367]), array([1.36960892]))
#[0.00291772]
#(array([0.00098437]), array([0.00430908]))
#[0.00573306]
#(array([0.38736148]), array([1.36900524]))
#[0.00289589]
    
"""
Statistics 

"""

alpha = 0.05

z = ann_average_error - linear_regression_average_error 

CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(CI)
print(p)

z = baseline_average_error-ann_average_error

CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(CI)
print(p)

z = baseline_average_error - linear_regression_average_error

CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(CI)
print(p)

#(array([0.47170315]), array([1.2998856]))
#[0.00092225]
#(array([-0.00116439]), array([0.00106571]))
#[0.92246333]
#(array([0.47153969]), array([1.30014774]))
#[0.00092509]







