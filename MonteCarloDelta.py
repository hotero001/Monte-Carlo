# Monte Carlo for European delta

import numpy as num

import math

# Step 1. Define parameters

S = 50.;           # Current asset value
h = 0.1;          # Size of perturbation of current asset value 
E = 52.;           # Strike price
T = 1.;            # Expiration time
r = 0.06;         # Risk free rate
sigma = 0.1;      # Volatility

M=100;            # Number of samples

# Step 2. Create a vector holding put values for each random sample

Vcall = num.zeros((M,1));
Vput = num.zeros((M,1));
Vhput = num.zeros((M,1))

for i in range(0,M):
  Sfinal = S*num.exp((r-0.5*sigma**2)*T+sigma*math.sqrt(T)*num.random.normal(0,1,1));
  Shfinal = (S+h)*num.exp((r-0.5*sigma**2)*T+sigma*math.sqrt(T)*num.random.normal(0,1,1));
  Vput[i]=num.exp(-r*T)*num.maximum(E-Sfinal,0); # Discounted payoff for a European put
  Vhput[i]=num.exp(-r*T)*num.maximum(E-Shfinal,0); # Discounted payoff for perturbed European put


# Step 3. Compute delta

aMput=num.mean(Vput);
aMhput=num.mean(Vhput);

bMput = num.std(Vput);
bMhput = num.std(Vhput);

delta = (aMhput-aMput)/h;
conf = [delta-1.96*(bMput+bMhput)/(h*math.sqrt(M)),delta+1.96*(bMput+bMhput)/(h*math.sqrt(M))];


# Step 4. Direct calculation 

d1 = (math.log(S/E)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T));
N1=0.5*(1+math.erf(d1/math.sqrt(2)));


# Step 5. Output Monte Carlo estimate of delta and true value confidence intervals and Black-Scholes values

print(delta);
print(conf);
print(N1-1)