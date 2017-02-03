'''
This contains the implementation of the policy evaluation algorithm on the gridworld in Example 4.1 of Sutton and Barto 2nd edition
'''

import numpy as np

'''Manually encode the states '''
states = range(15) #State 0 is the terminal state
moves = [[0,0,0,0],[0,1,2,5],[1,2,3,6],[2,3,3,7],[0,4,5,8],[1,4,6,9],[2,5,7,10],[3,6,7,11],[4,8,9,12],[5,8,10,13],[6,9,11,14],[7,10,11,0],[8,12,12,13],[9,12,13,14],[10,13,14,0]]
v = np.array([0 for x in range(15)],dtype=np.float32)

'''Perform iterative policy evaluation(refer to page 81 of edition 2 of Sutton and Barto for the algo) '''
epsilon = 0.00001
count = 1
while True:
    delta = 0
    for s in range(1,15):
        v_old = v[s]
        v[s] = np.sum(0.25*(-1.0+v[moves[s]]))
        delta = max(delta,abs(v[s]-v_old))
    if delta < epsilon:
        break
    count += 1

print 'Solution to Example 4.1'
print np.append(v,0).reshape(4,4)
print 'Took {} iterations to converge'.format(count)

'''Excercise 4.2 Part 1: Since the transitions from other states don't change, their value functions don't change too.'''
v_15 = 0
while True:
    v_old = v_15
    v_15 = np.sum(0.25*(-1+v[[12,13,14]]))+ 0.25*(-1+v_15)
    if abs(v_15-v_old)<epsilon:
        break
print '\n\n\nSolution for excercise 4.2 Part 1'
print 'Value function for states 0 to 14 remain unchanged'
print 'V(15)={}'.format(v_15)

'''Excercise 4.2 part 2: Now that transition affects state 13, it in a chain effect affects the other states as well '''
#Start with values initialised to the ones found before
v = np.append(v,v_15)
moves.append([12,13,14,15])
moves[13] = [9,12,14,15]


epsilon = 1e-10
count = 1
while True:
    delta = 0
    for s in range(1,16):
        v_old = v[s]
        v[s] = np.sum(0.25*(-1.0+v[moves[s]]))
        delta = max(delta,abs(v[s]-v_old))
    if delta < epsilon:
        break
    count += 1

print '\n\n\nSolution to Excercise 4.1 Part2'
print v
print 'Took {} iterations to converge'.format(count)
        

