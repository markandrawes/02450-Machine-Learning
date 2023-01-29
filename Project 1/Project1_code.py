import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import (boxplot, figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,title, legend,show, hist, ylim, xlim)

from scipy.linalg import svd


pd.set_option("display.max_rows", None, "display.max_columns", None)

Headers = ["RI" , "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]

attributeNames = ["RI" , "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

X = pd.read_csv("/Users/albert/Desktop/Glass.txt" , sep = "," , names = Headers)

classNames = ["building_windows_1", "building_windows_2", "vehicle_windows_1", "vehicle_windows_2", "containers", "tableware", "headlamps"]

print(X.describe())

X2 = X.copy()

del X2["Type"]

print(X2.corr())

X = X.to_numpy()

y = X[:,9]

X = X[:,:9]

N = len(y)
M = len(attributeNames)
C = len(classNames)

print(np.count_nonzero(np.isnan(X) != False))


"""

Plotting matrix of scatter plots

"""

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()


"""

Plotting histograms with normal distribution density 
function

"""

figure(figsize=(12,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)

for i in range(M):
    subplot(int(u),int(v),i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    
show()

"""

Plotting boxplots

"""

boxplot(X[:,0])
xticks([1], attributeNames[0:1])
ylabel('Refractive index')
title('Glass Identification dataset - boxplot')
show()

boxplot(X[:,1:4])
xticks(range(1,4),attributeNames[1:4])
ylabel('weight percent in corresponding oxides')
title('Glass Identification dataset - boxplot')
show()

boxplot(X[:,4])
xticks([1], attributeNames[4:5])
ylabel('weight percent in corresponding oxides')
title('Glass Identification dataset - boxplot')
show()

boxplot(X[:,5:9])
xticks(range(1,5),attributeNames[5:9])
ylabel('weight percent in corresponding oxides')
title('Glass Identification dataset - boxplot')
show()


"""

Checking for outliers

"""

print(np.count_nonzero(X >= 100))
print(np.count_nonzero(X < 0))



"""
Plotting variance explained
"""

N = len(X) 
mean = X.mean(axis=0)
# Subtract mean value from data
Y = (X - np.ones((N,1))*X.mean(axis=0))/(np.ones((N,1))*X.std(axis = 0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


"""
Plotting principal directions
"""

U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# We saw that the first 5 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b','p']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
   plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Glass Identification Dataset: PCA Component Coefficients')
plt.show()

print('PC2:')
print(V[:,1].T)

"""

Plotting projections

"""

# PCA by computing SVD of Y
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted

j = 4
i = 3

      
# Plot PCA of the data
f = figure()
title('Glass Identification data: PCA')
#Z = array(Z)

for c in range(1,C+1):
        
        # select indices belonging to class c:
        class_mask = y==c
        plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
        legend(classNames)
        xlabel('PC{0}'.format(i+1))
        ylabel('PC{0}'.format(j+1))
        # Output result to screen
show()

