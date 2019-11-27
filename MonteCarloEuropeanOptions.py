# Monte Carlo for European options

import numpy as num

import math

# Step 1. Define parameters

S = 50.;            # Current asset value
E = 52.;            # Strike price
T = 1.;             # Expiration time
r = 0.06;           # Risk free rate
sigma = 0.1;        # Volatility

M=100;              # Number of samples

# Step 2. Create a vector holding put and call values for each random sample

Vcall = num.zeros((M,1));
Vput = num.zeros((M,1));

for i in range(0,M):
  Sfinal = S*num.exp((r-0.5*sigma**2)*T+sigma*math.sqrt(T)*num.random.normal(0,1,1))
  Vput[i]=math.exp(-r*T)*num.maximum(E-Sfinal,0) # Discounted payoff for a European put
  Vcall[i]=math.exp(-r*T)*num.maximum(Sfinal-E,0)# Discounted payoff for a European call


# Step 3. Compute sample mean and standard deviation

aMput=num.mean(Vput);
bMput=num.std(Vput);

aMcall=num.mean(Vcall);
bMcall=num.std(Vcall);

# Step 4. Compute 95% confidence intervals for put and call values.

confPut = [aMput-1.96*bMput/math.sqrt(M),aMput+1.96*bMput/math.sqrt(M)];
confCall = [aMcall-1.96*bMcall/math.sqrt(M),aMcall+1.96*bMcall/math.sqrt(M)];

# Step 5. Direct calculation from Black-Scholes for put and call options

d1 = (math.log(S/E)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T));
d2=d1-sigma*math.sqrt(T);
N1=0.5*(1+math.erf(d1/math.sqrt(2)));
N2=0.5*(1+math.erf(d2/math.sqrt(2)));

bsCall=S*N1-E*math.exp(-r*T)*N2 
bsPut=E*math.exp(-r*T)*(1-N2)+S*(N1-1) 


# Step 6. Output Monte Carlo confidence intervals and Black-Scholes values
print(confPut)
print(bsPut)
print(confCall)
print(bsCall)

