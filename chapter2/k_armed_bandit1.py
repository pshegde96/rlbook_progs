'''
This program implements the stationary k-armed bandit problem using the epsilon-greedy algorithm
'''

import numpy as np
import random
import matplotlib.pyplot as plt



k = 10
N = 1000
eps = [0,0.01,0.1]
bandits = np.zeros((k,2))
Q_total = np.zeros((len(eps),N))
runs = 20000
for temp in range(runs):
	print temp
	#First define the destribution for the k bandits(gaussian distributed)
	for i in range(k):
		mu = (random.random()-0.5)*4 #random no between -2 and 2
		std = random.random()/2.0+0.5 #random no between 0.5 and 1
		bandits[i] = np.array([mu,std])
	r_av = list()
	for i in range(len(eps)):
		e = eps[i]
		R_av = [0]
		q = np.zeros(k)
		n_k = np.zeros(k)
		for n in range(1,N):
			explore_or_exploit = np.random.binomial(1,e)
			if explore_or_exploit == 0: #exploit
				choice = np.argmax(q)
				r = np.random.normal(bandits[choice,0],bandits[choice,1]) #reward
				Q_total[i,n] += 1.0/n*(r-Q_total[i,n])
				n_k[choice] += 1
				q[choice] += 1.0/n_k[choice]*(r-q[choice])

			else: #explore
				choice = np.random.choice(k)
				r = np.random.normal(bandits[choice,0],bandits[choice,1]) #reward
				Q_total[i,n] += 1.0/n*(r-Q_total[i,n])
				n_k[choice] += 1
				q[choice] += 1.0/n_k[choice]*(r-q[choice])
n = range(N)	
plt.plot(n,Q_total[0],'r')
plt.plot(n,Q_total[1],'b')
plt.plot(n,Q_total[2],'g')
plt.show()
