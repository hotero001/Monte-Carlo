# Monte Carlo for an arithmetic average Asian call option using a control variate.

import numpy as num
import math


# Step 1. Define parameters

S = 4.;             # Current asset value
E = 4.;             # Strike price
sigma = 0.25;       # Volatility
r = 0.03;           # Risk free rate
T = 1;             # Expiration time
Dt = 0.01;          # Length of time between observations of the asset
N = T/Dt;           # Total number of observations

M = 10000;          # Number of samples

# Step 2. Compute geometric average Asian call option exact value

sigsqT= (sigma**2)*T*(N+1)*(2*N+1)/(6*N*N);
muT = 0.5*sigsqT + (r-0.5*sigma**2)*T*(N+1)/(2*N);

d1 = (math.log(S/E)+(muT+0.5*sigsqT))/(math.sqrt(sigsqT));
d2 = d1 - math.sqrt(sigsqT);

N1 = 0.5*(1+math.erf(d1/math.sqrt(2)));
N2 = 0.5*(1+math.erf(d2/math.sqrt(2)));

geo =  math.exp(-r*T)*(S*math.exp(muT)*N1-E*N2);

# Step 3. Compute the ensemble of asset price trajectories

Spath=num.zeros((M,int(N)+1));      # Set up a placeholder M by N+1 array
Spath[:,0]=S;                       # Insert initial values

for i in range(0,M):
    for j in range(1,int(N)+1):
       Spath[i,j]=Spath[i,j-1]*math.exp((r-0.5*sigma**2)*Dt+sigma*math.sqrt(Dt)*num.random.normal(0,1,1)); 

# Step 4. Use brute force Monte Carlo to value the option

arithAvg = num.mean(Spath,axis=1);   # Arithmetic average of asset prices along each trajectory
pArith = math.exp(-r*T)*num.maximum(arithAvg-E,0);  # Compute payoff on each trajectory 
pMean = num.mean(pArith);           # Compute option value
pStd = num.std(pArith);             # Standard deviation of sampled option values
confIntMC = [pMean-1.96*pStd/math.sqrt(M), pMean+1.96*pStd/math.sqrt(M)]; # Confidence interval for our estimation of the option value.

print(pMean);                       # Output estimated option value
print(confIntMC);                   # Output CI for option value

# Step 5. Use the geometric average Asian call option value as a control variate

geoAvg = num.exp((1/(N+1))*num.sum(num.log(Spath),axis=1));    # Geometric average of asset prices along each trajectory. Note that we use num.exp() instead of math.exp() because the latter does not accept arrays as arguments.
pGeo = math.exp(-r*T)*num.maximum(geoAvg-E,0);          # Compute payoff on each trajectory. 
z = pArith + geo - pGeo;               # Control variate 
zMean = num.mean(z);                   # Compute option value
zStd = num.std(z);                     # Standard deviation of sampled option values
confIntCVMC = [zMean-1.96*zStd/math.sqrt(M), zMean+1.96*zStd/math.sqrt(M)]; # Confidence interval for our estimation of the option value.

print(zMean);                       # Output estimated option value
print(confIntCVMC);                 # Output CI for option value
