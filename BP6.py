import numpy as np

N, D_in, H, D_out = 5, 5, 10, 5

# x = np.array([ [0,0],[0,1],[1,0],[1,1] ])
# y = np.array([[0,0,0,1]]).T
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
x = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1],
            ])
y = x.copy()

# syn0 = np.random.random((2,4))
# syn1 = np.random.random((4,1))
syn0 = np.random.randn(D_in, H)
syn1 = np.random.randn(H, D_out)

lean_rate=5
for j in range(1000):
    l1 = 1/(1+np.exp(-(np.dot(x,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))    

    print(((l2-y)**2).sum())
    l2_delta = (l2-y)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))   
    syn1 -= lean_rate*l1.T.dot(l2_delta)
    syn0 -= lean_rate*x.T.dot(l1_delta)


l1 = 1/(1+np.exp(-(np.dot(x,syn0))))
l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))

print(l2)
