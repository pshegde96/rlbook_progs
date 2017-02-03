import numpy as np
from scipy.stats import poisson
import operator #Used to get max key of a dictionary

'''Setup the parameters of the problem '''
prequest1 = poisson.pmf(range(21),3)
prequest2 = poisson.pmf(range(21),4)
preturn1 = poisson.pmf(range(21),3)
preturn2 = poisson.pmf(range(21),2)
max_cars = 20
max_move = 5
rent_reward = 10 
move_reward = -2
gamma = 0.9 #discount rate

'''Set up the initial policies and value functions '''
val_fn = np.zeros((21,21))
policy = np.zeros((21,21))

'''Setup the rewards for the state transitions
s11: This is the state of location1 after the action of moving cars overnight has been made
s12: This is the state of location2 after the end of one day
Likewise for location 2
'''
rewards = np.zeros((21,21,21,21))
rewards1 = np.zeros((21,21))
rewards2 = np.zeros((21,21))
trans_prob = np.zeros((21,21,21,21))
trans1 = np.zeros((21,21))
trans2 = np.zeros((21,21))

for s11 in range(21):
    for s12 in range(21):
        req = np.arange(0,s11+1)
        ret = s12 - req
        trans1[s11,s12] = np.sum(prequest1[req]*preturn1[ret])
        rewards1[s11,s12] = np.sum(prequest1[req]*preturn1[ret]*rent_reward*req)/trans1[s11,s12]
trans1 = trans1/np.sum(trans1) #normalise probabilities
for s21 in range(21):
    for s22 in range(21):
        req = np.arange(0,s21+1)
        ret = s22 - req
        trans2[s21,s22] = np.sum(prequest2[req]*preturn2[ret])
        rewards1[s21,s22] = np.sum(prequest2[req]*preturn2[ret]*rent_reward*req)/trans2[s21,s22]
trans2 = trans2/np.sum(trans2) #normalise probabilities
for s11 in range(21):
    for s12 in range(21):
        for s21 in range(21):
            for s22 in range(21):
                trans_prob[s11,s21,s12,s22] = trans1[s11,s12]*trans2[s21,s22]
                rewards[s11,s21,s12,s22] = rewards1[s11,s12] + rewards2[s21,s22]

                
'''Perform Policy iteration '''
count = 0
while True:

    epsilon = 1e-3
    #Policy Evaluation
    while True:
        delta = 0
        for s1 in range(21):
            for s2 in range(21):
                v_old = val_fn[s1,s2]
                val_fn[s1,s2] = np.sum(trans_prob[s1-policy[s1,s2],s2+policy[s1,s2],:,:]*(-2*abs(policy[s1,s2])+rewards[s1-policy[s1,s2],s2+policy[s1,s2],:,:]+gamma*val_fn))
                delta = max(delta,abs(val_fn[s1,s2]-v_old))
        if delta < epsilon:
            break
    poliold = policy.copy()    
    #Policy improvement
    policy_stable = True
    for s1 in range(21):
        for s2 in range(21):
            old_action = policy[s1,s2]

            #actions of moving cars from loc1 to loc2
            pol_val = dict()
            for a in range(min(5,s1,20-s2)+1):
                pol_val[a] = np.sum(trans_prob[s1-a,s2+a,:,:]*(-2*abs(a)+rewards[s1-a,s2+a,:,:]+gamma*val_fn))
            for a in range(1,min(5,s2,20-s1)+1):
                pol_val[-a] = np.sum(trans_prob[s1+a,s2-a,:,:]*(-2*abs(a)+rewards[s1+a,s2-a,:,:]+gamma*val_fn))
            
            policy[s1,s2] = max(pol_val.iteritems(), key=operator.itemgetter(1))[0]
            if old_action != policy[s1,s2]:
                policy_stable = False
    count += 1
    print count
    if policy_stable == True:
        break

print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
              for row in np.flipud(policy).astype(np.int8)]))




