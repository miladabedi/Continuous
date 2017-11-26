import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance
import csv
import Action_Value
import imp
import time
from sklearn.linear_model import LinearRegression 
imp.reload(Action_Value)
def Evaluate_Action_Value_Function(Weights,Experiences):
    
    Num_Exp=np.shape(Experiences)[0]
    Error=np.zeros(Num_Exp)
    
    for i in range(Num_Exp-1):
        Feature_t=np.concatenate([Experiences[i][0],Experiences[i][1]])
        Feature_t[-1]=np.log(Feature_t[-1])
        Feature_t_Plus_1=np.concatenate([Experiences[i][3],Experiences[i][1]])
        Feature_t_Plus_1[-1]=np.log(Feature_t_Plus_1[-1])
        
        
        
        
        Predicted_Reward=Action_Value.Linear_Action_Value(Feature_t,Weights)-Action_Value.Linear_Action_Value(Feature_t_Plus_1,Weights)
        Error[i]=(Experiences[i][2]-Predicted_Reward)/Experiences[i][2]
    return Error



'Load Experience'

Exp=np.load('Exp.npy').tolist()

'initialize parameteres'
W=np.zeros(8)
Elig=W
delta_W=0
Memory_Replay_Depth=100


Err=np.zeros(100)

for j in range(Memory_Replay_Depth):
    
#    for i in range(np.shape(Exp)[0]):
    for i in np.nditer(aran):
        delta_W,Elig,Error=Action_Value.Linear_Action_Value_Function_Weight_Updator_TD_Lambda(W,Exp[i],Elig,0)
        
        W+=delta_W
#        print(delta_W)
#        print(np.sum(np.abs(delta_W/W)),Exp[i][1][3])
#        print('------------------------------------------------------')
        
#        Err[i]=Error/Exp[i][2]
#plt.hist(Err,100)
print(np.mean(np.abs(Err)))
#print(np.sum(abs(Evaluate_Action_Value_Function(W,Exp[100:600]))))
#time.sleep(10)
plt.hist(Evaluate_Action_Value_Function(W,Exp[101:200]),100)
a=Evaluate_Action_Value_Function(W,Exp[101:200])
print(np.mean(Evaluate_Action_Value_Function(W,Exp[101:200])))



#aran=np.random.randint(0,300,100)
Features=[]
Rewards=[]
import random
random.shuffle(Exp)
for i in range(np.shape(Exp)[0]):
    Feature_t=np.concatenate([Exp[i][0],Exp[i][1]])
#    Feature_t[-1]=np.log(Feature_t[-1])
    Features.append(Feature_t)
    Rewards.append(Exp[i][2])
X=np.array(Features)
y=np.array(Rewards)
y.ravel
mdl=LinearRegression()
mdl.fit(X[0:6],y[0:6])
pred=mdl.predict(X)
print(np.mean(np.abs((pred[50:]-y[50:])/y[50:])))
plt.hist(np.abs((pred[50:]-y[50:])/y[50:])*100,1000)
mdl.coef_
