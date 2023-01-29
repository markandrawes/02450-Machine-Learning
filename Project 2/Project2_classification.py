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

Headers = ["RI" , "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]

attributeNames = ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

X = pd.read_csv("/Users/albert/Desktop/Glass.txt" , sep = "," , names = Headers)

classNames = ["building_windows_1", "building_windows_2", "vehicle_windows_1", "vehicle_windows_2", "containers", "tableware", "headlamps"]

X = X.to_numpy()

y = X[:,9]

y = y-1

X = X[:,:9]

N, M = X.shape

X = stats.zscore(X)

C = len(classNames)

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
logistic_regression_average_error = np.empty((K1, 1))
logistic_regression_test_error = np.zeros(X.shape[0])
ann_average_error = np.empty((K1, 1))
ann_test_error = np.zeros(X.shape[0])
mu = np.empty((K1, M))
sigma = np.empty((K1, M))
w_rlr = np.empty((M+1, K1))
opt_lambdas_rlr = np.empty((K1, 1))

n_replicates = 1 # number of networks trained in each k-fold
n_hidden_units = [1,2,3,4,5,6,7,8,9,10] # number of hidden units in the single hidden layer
#n_hidden_units = [1,5]

max_iter = 10000 # Train for a maximum of 10000 steps, or until convergence (see help for the
loss_fn = torch.nn.CrossEntropyLoss() # notice how this is now a mean-squared-error loss
opt_hidden_units = np.empty((K1, 1))

y_hat_logistic = []
y_hat_ANN = []
y_hat_baseline = []

#loss function:
    
loss_fn = torch.nn.CrossEntropyLoss()    

for k, (train_index, test_index) in enumerate(CV_outer.split(X,y)):
    
    print(k)

    # extract training and test set for current CV fold (outer fold)
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    #Baseline
    baseline_class_prediction = int(stats.mode(y_train)[0][0])
    
    baseline_test_prediction = int(stats.mode(y_train)[0][0])*np.ones(len(y_test))
    
    y_hat_baseline.append(baseline_test_prediction)
    
    baseline_average_error[k] = np.sum(y_test != baseline_class_prediction) / len(y_test)
    #baseline_test_error[test_index] = (y_test != baseline_class_prediction)
    
    #Logistic regression classifier and ANN
    
    for train_index_ann_lr, test_index_ann_lr in CV_inner.split(X_train, y_train):
        
        #Logistic regression
        
        X_train_lr = X[train_index_ann_lr,:]
        y_train_lr = y[train_index_ann_lr]
        X_test_lr = X[test_index_ann_lr,:]
        y_test_lr = y[test_index_ann_lr]
        
        lambda_interval = np.power(10.,range(-6,10))
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))    
        coefficient_norm = np.zeros(len(lambda_interval))
        
        for s in range(0, len(lambda_interval)):
            
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[s] )
        
            mdl.fit(X_train_lr, y_train_lr)
    
            y_train_est = mdl.predict(X_train_lr).T
            y_test_est = mdl.predict(X_test_lr).T
        
            train_error_rate[s] = np.sum(y_train_est != y_train_lr) / len(y_train_lr)
            test_error_rate[s] = np.sum(y_test_est != y_test_lr) / len(y_test_lr)
    
            w_est = mdl.coef_[0] 
            coefficient_norm[s] = np.sqrt(np.sum(w_est**2))
           
        
        #ANN
        errors = np.zeros((len(n_hidden_units), K2))  # make a list for storing generalizaition error for each model

        i = 0
    
        #X_train_ann = torch.Tensor(X_train_1)
        #y_train_ann = torch.Tensor(y_train_1)
        #X_test_ann = torch.Tensor(X_test_1)
        #y_test_ann = torch.Tensor(y_test_1)

        # Model loop
        j = 0
        
        for n in n_hidden_units:
            
            
            inner_model = lambda: torch.nn.Sequential(
                                            torch.nn.Linear(M, n), #M features to H hiden units
                                            torch.nn.ReLU(), # 1st transfer function
                                            # Output layer:
                                            # H hidden units to C classes
                                            # the nodes and their activation before the transfer 
                                            # function is often referred to as logits/logit output
                                            torch.nn.Linear(n, C), # C logits
                                            # To obtain normalised "probabilities" of each class
                                            # we use the softmax-funtion along the "class" dimension
                                            # (i.e. not the dimension describing observations)
                                            torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit 
                                            
                                            )
                
                
            # Train the net on training data
            net, _, _ = train_neural_net(inner_model, loss_fn,
                             X=torch.tensor(X_train_lr, dtype=torch.float),
                             y=torch.tensor(y_train_lr, dtype=torch.long),
                             n_replicates=n_replicates,
                             max_iter= max_iter)

            # Determine probability of each class using trained network
            softmax_logits = net(torch.tensor(X_test_lr, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
            # Determine errors
            e = np.sum((y_test_est != y_test_lr))/(len(y_test_lr))
            #print('Proportion of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))
            errors[j][i] = e

            j += 1
        i += 1

    opt_hidden_units[k] = n_hidden_units[np.argmin(np.mean(errors, axis=1))]
    opt_hidden_unit = opt_hidden_units[k][0].astype(int)    
    
    # Compute error for best performing model for the outer loop
    outer_model = lambda: torch.nn.Sequential(
                                   torch.nn.Linear(M, opt_hidden_unit), #M features to H hiden units
                                   torch.nn.ReLU(), # 1st transfer function
                                   # Output layer:
                                   # H hidden units to C classes
                                   # the nodes and their activation before the transfer 
                                   # function is often referred to as logits/logit output
                                   torch.nn.Linear(opt_hidden_unit, C), # C logits
                                   # To obtain normalised "probabilities" of each class
                                   # we use the softmax-funtion along the "class" dimension
                                   # (i.e. not the dimension describing observations)
                                   torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit 
                                   
                                   )
    
    # Train the net on training data
    net, _, _ = train_neural_net(outer_model, loss_fn,
                     X=torch.tensor(X_train, dtype=torch.float),
                     y=torch.tensor(y_train, dtype=torch.long),
                     n_replicates=n_replicates,
                     max_iter= max_iter)

    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    
    y_test_est_ann = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
    
    y_hat_ANN.append(y_test_est_ann)
    
    # Determine errors
    #se = (y_test_est_ann.float() - y_test_ann.float()) ** 2  # squared error
    #mse = (sum(se).type(torch.float) / len(y_test_ann)).data.numpy().mean()  # mean
    e = np.sum((y_test_est_ann != y_test))/(len(y_test))
    
    ann_average_error[k] = e
    
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
        
    opt_lambdas_rlr[k] = opt_lambda
        
    logistic_regression_average_error[k] = np.average(test_error_rate)
        
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
        
    mdl.fit(X_train, y_train)
        
    y_test_estimate_logistic = mdl.predict(X_test)
        
    y_hat_logistic.append(y_test_estimate_logistic) 

output_data = np.hstack((
    np.arange(K1).reshape(K1,1) + 1,
    opt_hidden_units,
    ann_average_error,
    opt_lambdas_rlr,
    logistic_regression_average_error,
    baseline_average_error
))

print(tabulate(
    output_data,
    headers=['i','ann_h','ann_err','lr_lambda','lr_err','base_err']))

#i    ann_h    ann_err    lr_lambda    lr_err    base_err
#---  -------  ---------  -----------  --------  ----------
#1        9  0.0526316    1e-08      0.24        0.636364
#2        5  0.0526316    1e-08      0.48        0.636364
#3        7  0.0526316    0.13895    0.510526    0.636364
#4        5  0.0526316    1e-08      0.374737    0.5
#5        4  0.0526316    1e-08      0.437895    0.761905
#6        1  0.0526316    0.0542868  0.350526    0.666667
#7       10  0.0526316    9.54095    0.349474    0.714286
#8        2  0.0526316    9.54095    0.346316    0.809524
#9       10  0.0526316    9.54095    0.392632    0.52381
#10       1  0.0526316  100          0.386316    0.857143


"""
Statistics for classification
"""

yhat_log= np.concatenate(y_hat_logistic)
yhat_ANN = np.concatenate(y_hat_ANN)
yhat_baseline = np.concatenate(y_hat_baseline)
y_true = y

#yhat_ANN = np.reshape(y_hat_ANN,-1)
#yhat_baseline = np.reshape(y_hat_baseline,-1)
#y_true = np.reshape(y,-1)


alpha=0.05
print("")
print("Logistic vs ANN")
[thetahat_logANN, CI_logNN, p_logANN] = mcnemar(y_true, yhat_log, yhat_ANN, alpha=alpha)
print("")
print("Baseline vs Logistic")
[thetahat_logbase, CI_logbase, p_logbase] = mcnemar(y_true, yhat_log, yhat_baseline, alpha=alpha)
print("")
print("ANN vs Baseline")
[thetahat_ANNbase, CI_ANNbase, p_ANNbase] = mcnemar(y_true, yhat_ANN, yhat_baseline, alpha=alpha)

"""
Logistic regression

"""

Headers = ["RI" , "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]

attributeNames = ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

X = pd.read_csv("/Users/albert/Desktop/Glass.txt" , sep = "," , names = Headers)

classNames = ["building_windows_1", "building_windows_2", "vehicle_windows_1", "vehicle_windows_2", "containers", "tableware", "headlamps"]

X = X.to_numpy()

N, M = X.shape

X = stats.zscore(X)

y = X[:,9]

X = X[:,:9]

y = y-1



log_model_with_reg = LogisticRegression(penalty='l2', C = 1/0.01, solver = 'lbfgs', fit_intercept = True) # remember to select the most-commonly found optimal lambda value
log_model_with_reg = log_model_with_reg.fit(X, y)

print('Weights for LogReg model with regularization:')
print('{:>20} {:>20}'.format('Intercept', str(np.round(log_model_with_reg.intercept_[0],3))))
for m in range(M):
    print('{:>20} {:>20}'.format(attributeNames[m], str(np.round(log_model_with_reg.coef_[0][m],3))))






#i    ann_h    ann_err    lr_lambda    lr_err    base_err
#---  -------  ---------  -----------  --------  ----------
#1        5   0.590909        1      0.578947    0.863636
#2        9   0.272727        1      0.490132    0.590909
#3        9   0.318182        1e-05  0.417763    0.727273
#4        2   0.363636        0.01   0.509868    0.545455
#5        2   0.428571        10     0.467105    0.857143
#6        2   0.428571        0.01   0.5625      0.761905
#7        5   0.238095        0.01   0.463816    0.666667
#8        4   0.380952        1e-06  0.585526    0.761905
#9        3   0.238095        1e-06  0.496711    0.571429
#10       1   0.333333        10     0.588816    0.666667

#Logistic vs ANN
#Result of McNemars test using alpha= 0.05ª
#Comparison matrix n
#[[ 42.  17.]
#[ 15. 140.]]
#Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (-0.04233270951234791, 0.0609996664028738)
#p-value for two-sided test A and B have same accuracy (exact binomial test): p= 0.860050065908581

#Baseline vs Logistic
#Result of McNemars test using alpha= 0.05
#Comparison matrix n
#[[ 25.  34.]
#[ 52. 103.]]
#Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (-0.1677710820895406, -0.00013726000401703686)
#p-value for two-sided test A and B have same accuracy (exact binomial test): p= 0.04915259746138927

#ANN vs Baseline
#Result of McNemars test using alpha= 0.05
#Comparison matrix n
#[[ 27.  30.]
#[ 50. 107.]]
#Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (-0.17389128022710543, -0.01241694579859598)
#p-value for two-sided test A and B have same accuracy (exact binomial test): p= 0.03299261842647619


