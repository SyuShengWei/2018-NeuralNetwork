
# coding: utf-8

# In[8]:


import numpy as np


# In[42]:


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
    #return(np.tanh(x))
def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))
    #return 1-(x**2)


# In[43]:


X = np.array(([0.05,0.05],
              [0.05,0.95],
              [0.95,0.05],
              [0.95,0.95]), dtype = float)
D = np.array(([0.05],
              [0.95],
              [0.95],
              [0.05]), dtype =float)


# In[44]:


class BPN(object):
    
    def __init__(self,eta,alpha):
        self.eta = eta
        self.alpha = alpha
        
        self.inputSize  = 3
        self.hiddenSize = 3
        self.outputSize = 1
        
        self.Wih = np.random.rand(self.inputSize, self.hiddenSize-1)
        self.Whj = np.random.rand(self.hiddenSize, self.outputSize)
        #self.Wih = np.ones([self.inputSize, self.hiddenSize-1])
        #self.Whj = np.ones([self.hiddenSize, self.outputSize])
        
        self.delta_Wih_last = 0
        self.delta_Whj_last = 0 
        
    def inputBias(self,Xk):
        return  np.append(np.array([1.0],dtype =float),Xk)
    def hiddenBias(self,Zk):
        return  np.append(np.array([1.0],dtype =float),Zk)
    
    def forward(self,Xk):
        self.Xk = self.inputBias(Xk)
        self.Zh = self.Xk.dot(self.Wih)
        self.SZh = sigmoid(self.Zh)
        self.Zh = self.hiddenBias(self.SZh)
        self.Yj = self.Zh.dot(self.Whj)
        self.SYj = sigmoid(self.Yj)
    
    def backward(self, Xk, Dk):
        self.deltaJ = (Dk-self.SYj)*sigmoid_prime(self.Yj)
        self.deltaWhj = (self.eta * self.deltaJ[0] * sigmoid(self.Xk.T)).reshape([self.hiddenSize,1])
        
        self.deltaH = np.ones([self.hiddenSize,1])
        for i in range(0,self.hiddenSize):
            sum_up = 0
            for j in range(0,self.outputSize):
                sum_up += self.deltaJ[j]*self.Whj[i][j]
            self.deltaH[i][0] = sum_up*sigmoid_prime(self.Zh[i])
        
        
        self.deltaWih = np.ones([self.inputSize,self.hiddenSize-1])
        for i in range(0,self.inputSize):
            for h in range(1,self.hiddenSize):
                self.deltaWih[i][h-1] = self.eta * self.deltaH[h] * self.Xk[i]
                
        self.Wih += (self.deltaWih )
        self.Whj += (self.deltaWhj )
        
        self.delta_Wih_last = self.deltaWih
        self.delta_Whj_last = self.deltaWhj


# In[45]:


bpn = BPN(0.1,0.01)


# In[46]:


E_average = [100000]
while ( abs(E_average[0]) >= 0.001):
    E_sum = 0
    for i in range(0,4):
        bpn.forward(X[i])
        #print(X[i],bpn.SYj,bpn.Yj)
        bpn.backward(X[i],D[i])
        #print(X[i],D[i])
        E_sum += ((D[i]-bpn.SYj)**2) /2
    E_average = E_sum/4
    #print(bpn.Whj)
    print(E_average)


# In[47]:


for i in range(4):
    bpn.forward(X[i])
    print(bpn.SYj)

