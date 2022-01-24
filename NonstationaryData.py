import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
import pandas as pd

n_noise = 1
n_Drift = 10
random.seed(1234)

T = 1000
dt = 0.05

def D1(x,t):

#    return(- 6*x**1 + x**2  + 1.*np.sin(t)   ) # + 3*x**3.-2*x**4
    return(-0.5*x**3. + 0.5*t)

def D2(x,t):
    return(0.1)

x0 = 0.5
X = np.zeros(T)
X[0] = x0
t = np.arange(T)*dt


for i in range(1,T):
    X[i] = X[i-1] + D1(X[i-1], t[i-1]) *dt + np.sqrt(D2(X[i-1], t[i-1]))*np.sqrt(dt)*random.gauss(0,1)



Data = np.empty((T,2))
Data[:,0] = t
Data[:,1] = X

###
plt.plot(t, X)
plt.savefig("NonstationaryData")
plt.show()



####

## Define all the functions

#@jit
def poly(x,sigma):
    #x_vec=np.array([1,x[0],x[1],x[0]*x[0],x[1]*x[0],
    #                x[1]*x[1],x[0]*x[0]*x[0],x[1]*x[0]*x[0],x[1]*x[1]*x[0],x[1]*x[1]*x[1],
    #               x[0]**4, x[0]**3 * x[1], x[0]**2 * x[1]**2, x[0]*x[1]**3, x[1]**4],
    #                dtype=object)

	# decoupled dynamics:
    x_vec=np.array([ np.sin(x[0]), np.cos(x[0]),
                    x[0], x[0]**2., x[0]**3., x[0]**4., # only time x[0] = t
					x[1], x[1]**2., x[1]**3., x[1]**4.],   # only observable x[1] = x
					dtype=object)


    #x_vec = np.array([1,x[0], x[1], x[0]*x[0], x[1]*x[0],
    #                   x[1]*x[1]], ) # simple form
    return np.dot(sigma,x_vec)



#@jit
def D1(sigma,x):
    # Make use of the fact that here, x_1 = t and hence dx_1/dt = 1
    sigma_d1 = sigma[n_noise:] # without noise parameters
    return poly(x.T, sigma_d1)


#@jit
def D2(alpha,x):
    return alpha[0]*np.ones(x.shape[0])


def LogPseudoPrior(alphas, TS):
    # takes coefficients for the Polynomials of the Time Series TS and the Time Series itself
    # returns prior for the given polynomial coefficients
    # if absolute(data) range is smaller than 1: larger polynomials should be less likely
    # else: higher polynomials more likely than smaller ones

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # In this code:
    # alpha[7, 8, 9, 10] are polynomial coefficients of x

    Coefficient = np.mean(abs(TS)) # characteristic value for the Time Series Data

    Output = np.empty(len(alphas)) #

    for i in range(len(alphas)):
        # i=0 is linear polynomials, hence exponent always i+1
        Output[i] = - np.abs(alphas[i]) / Coefficient**np.sqrt(i+1)
    return(Output.mean()*(Coefficient<1.))


#  Log Likelihood and negative logL

def log_likelihood(alpha,x,dt):
    # alpha is the current set of parameters
    # x is the entire data set N x 2
    # dt is the time difference

    log_like = 0 # initial value of sum

    if alpha[0]>0  : # noise must be positive

        dx = x[1:,:]-x[:-1,:]

        # only the x[:,1] component is the actual time series
        d1 = dx[:,1] - D1(alpha,x)[:-1]*dt
        d2 = D2(alpha,x)[:-1]
        d2_inv = d2**(-1.)
        log_like = ( -0.5*np.log(d2) - 0.5 * d2_inv * d1**2./dt -np.log(dt)).sum()

        # LogPseudoPrior: x[:,1] is the real TS, alphas[7:] the four polynomial coefficients
        return log_like + 0.001*LogPseudoPrior(alpha[7:], x[:,1])
    else:
        return -np.inf


def neg_log_likelihood(alpha,x,dt): #L Threshold Lambdac
    return -1*log_likelihood(alpha,x,dt)


# NEVER cut off all noise terms!
def cutIndex(Parameters, Threshold):
    Index = (np.abs(Parameters) < Threshold)
    Index[0:n_noise] = False
    return(Index)

def second_neg_log_likelihood(Coeff, Index,x,dt):
    # Index: Index of those coefficients which are set to 0: Boolean
    Index = cutIndex(Coeff,L_thresh)
    Coeff[Index] = 0
    return -1*log_likelihood(Coeff,x,dt)


# BIC as goodness criterion for a threshold value

def BIC(alpha,x,dt,L): # mit Lambda Threshold

    logi = np.abs(alpha)>L # which are larger than Lambda?
    logi[0:n_noise] = True  # noise is always included
    return np.log(x[:,0].size)*np.sum(  logi ) - 2*log_likelihood(alpha, x,dt )


# Calculate BIC in the Loop with thresholding

def Loop(x, dt, L, a_Ini):
    # estimates alpha parameters based on starting values a_Ini for a given threshold L
    a_hat = optimize.minimize(neg_log_likelihood, a_Ini,args=(x,dt)) # max likelihood
    Estimation = a_hat["x"]
    Estimation[n_noise:] =  Estimation[n_noise:] * ( abs(Estimation[n_noise:] ) > L )

    for i in np.arange(0,n_Cut):
        Cut = (np.abs(Estimation)<L) # Boolean of the values that are cut off
        # second optimization with maxLikelEstimator as start:
        a_hat = optimize.minimize(second_neg_log_likelihood,Estimation,args = (Cut,x,dt))
        Estimation = a_hat["x"]
        Estimation[n_noise:] =  Estimation[n_noise:] * ( abs(Estimation[n_noise:] ) > L )
    return(Estimation)




######
# For consistency with past codes: call Data x
x = Data

# Set up variables for the hyperparameter search on threshold

n_Cut = 5 # Number of reiterating
hp1 = np.arange(0.00,0.31, 0.05) # list of possible thresholds
n_Iteration = len(hp1) # Number of Hyperparameter search iterations
score = np.empty(n_Iteration) # score for Hyperparameters



TestAl = np.ones(n_noise+n_Drift)   # sample parameters to start the search




#### Initial Guess
Start = np.ones(n_noise+n_Drift)
Start[0] = 0.01

Optim = optimize.minimize(neg_log_likelihood,Start,args=(x,dt))
print(Optim["x"])

###

AlphaList = np.empty((n_Iteration, n_noise + n_Drift))

for i in range(n_Iteration):
    L_thresh = hp1[i]
    estimate = Loop(x, dt, hp1[i], Optim["x"])
    AlphaList[i,:] = estimate
    score[i] = BIC(estimate, x, dt, hp1[i])
    print(estimate)
    print(hp1[i], score[i])


np.savetxt('Synthetic_Alpha_Done.txt',AlphaList)
d = {'Threshold': hp1, 'BIC': score}
df = pd.DataFrame(data=d)
print(df)
df.to_csv("Synthetic_Score_BIC.csv")

###
BestThreshold = df.loc[df['BIC'].idxmin()]['Threshold']

print("Best threshold is", BestThreshold)
BestCoeff = AlphaList[df["Threshold"] == BestThreshold,:]
print("Corresponding coefficients are", BestCoeff[0][0].round(5), "for the noise and ", BestCoeff[0][1:].round(2), "for the drift terms.")

### Evaluate the polynomial in x
xmin, xmax = min(X), max(X)
xrange = np.linspace(xmin, xmax, 100)
# true polynomial
def TruePoly(x):
    return(-0.5*x**3)
# Estimated polynomial coefficients
def PolyHat(x):
    vec_x = np.array([ x, x**2, x**3, x**4])
    return(np.dot(BestCoeff[0][7:], vec_x))
plt.hist(X, normed=True, alpha = 0.2, label="Data")
plt.plot(xrange, TruePoly(xrange), label="True Polynomial")
plt.plot(xrange, PolyHat(xrange), label="Estimation")
plt.legend()
plt.show()



