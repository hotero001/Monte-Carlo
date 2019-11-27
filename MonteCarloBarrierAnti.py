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

M = 2500;            # Number of samples
N = 100.;           # Number of steps taken for each trajectory
dt = T/N;           # Step size

# Step 2. Generate sample trajectories of the asset and corresponding option values

V = num.zeros((M,1));
Vanti = num.zeros((M,1));
Svals=num.zeros((N,1));
SvalsAnti=num.zeros((N,1));

for i in range(0,M):
  samples = num.random.normal(0,1,int(N))
  Svals[0]=S;
  SvalsAnti[0]=S;
  V2 = 0;
  
  # Step 2.1
  # Generate asset trajectory
  for j in range(0,int(N)-1):
    Svals[j+1] = Svals[j]*num.exp((r-(sigma**2)/2)*dt+sigma*math.sqrt(dt)*samples[j]);


  # Check if barrier achieved and apply payoff if not
  Smin=num.amin(Svals);
  if Smin > B: V[i]=num.exp(-r*T)*num.maximum(Svals[N-1]-E,0);

  
  # Generate antithetic asset trajectory
  for j in range(0,int(N)-1):
    SvalsAnti[j+1] = SvalsAnti[j]*num.exp((r-(sigma**2)/2)*dt-sigma*math.sqrt(dt)*samples[j]);


  # Check if barrier achieved and apply payoff if not
  Smin=num.amin(SvalsAnti);
  if Smin > B: V2=num.exp(-r*T)*max(SvalsAnti[N-1]-E,0);
  
  Vanti[i]=(V[i]+V2)/2;
    


# Step 3. Value the option by taking expectation and output a 95% confidence interval

aManti = num.mean(Vanti);
bManti=num.std(Vanti);
conf = [aManti - 1.96*bManti/math.sqrt(M),aManti + 1.96*bManti/math.sqrt(M)]

print(aManti);
print(conf);
  