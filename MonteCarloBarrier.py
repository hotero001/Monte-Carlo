# Monte Carlo for an down-and-out call option

import numpy as num

import math

# Step 1. Define parameters

S = 55.;            # Current asset value
E = 52.;            # Strike price
T = 1.;             # Expiration time
r = 0.06;           # Risk free rate
sigma = 0.1;        # Volatility
B = 49.;            # Barrier value

M = 5000;           # Number of samples
N = 100.;           # Number of steps taken for each trajectory
dt = T/N;           # Step size

# Step 2. Generate sample trajectories of the asset and corresponding option values

V=num.zeros((M,1));
Svals=num.zeros((N,1));

for i in range(0,M):
  samples = num.random.normal(0,1,int(N))
  Svals[0]=S;
  # Generate asset trajectory
  for j in range(0,int(N)-1):
    Svals[j+1] = Svals[j]*num.exp((r-(sigma**2)/2)*dt+sigma*math.sqrt(dt)*samples[j]);
  # Check if barrier achieved and apply payoff if not
  Smin=num.amin(Svals);
  if Smin > B:
    V[i]=num.exp(-r*T)*num.maximum(Svals[N-1]-E,0);
  

# Step 3. Value the option by taking expectation and output a 95% confidence interval

aM = num.mean(V);
bM=num.std(V);
conf = [aM - 1.96*bM/math.sqrt(M),aM + 1.96*bM/math.sqrt(M)];

print(aM);
print(conf);

  