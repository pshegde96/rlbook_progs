import numpy as np
import matplotlib.pyplot as plt

'''Problem parameters '''
ph = 0.4
max_dollars = 100
rew = 1 #After a win i.e., after reaching state 100

val_fn = np.zeros(101)
val_fn[100] = 1.0


'''Perform Value Iteration'''
epsilon = 1e-5
count = 0
while True:
    delta = 0
    for s in range(1,100):
        v_old = val_fn[s]
        q = {}
        for a in range(0,min(s,100-s)+1):
            q[a] = ph*val_fn[s+a] + (1-ph)*val_fn[s-a]
        val_fn[s] = q[max(q,key=q.get)]
        delta = max(delta,abs(val_fn[s]-v_old))

    count +=1
    print count

    if count == 32:
        break
plt.plot(val_fn)
plt.show()

'''Optimal Policies '''
policy = []
for s in range(1,100):
    actions = []
    best = 0
    for a in range(0,min(s,100-s)+1):
        q = ph*val_fn[s+a] + (1-ph)*val_fn[s-a]
        if q > best:
            actions = [a]
            best = q
        elif q == best:
            actions.append(a)
    policy.append(actions[0])
    print '{} : {}'.format(s,actions)
plt.plot(policy)
plt.show()
