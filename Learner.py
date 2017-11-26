fimport numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance
import csv
import Action_Value
import imp
import time
from sklearn.linear_model import LinearRegression 
import random

imp.reload(Action_Value)
#
Exp=np.load('Exp.npy').tolist()
#Exp=Exp1
#random.shuffle(Exp)
#temp=Exp[0:250]
#random.shuffle(temp)
#Exp[0:250  ]=temp
Learning_Rate=0.005
Theta=np.zeros([3,4])
#Theta=np.zeros([3,3])/4
W_T=np.zeros([500,9])
for j in range(1):
    for i in np.arange(50):
        Weights=np.zeros(9)


        Theta=np.ravel(Theta)
        Theta+=Learning_Rate*Action_Value.Gaussian_Policy_Score_Function(Exp[i],Theta)*(Action_Value.Linear_Action_Value(Action_Value.S_A_Feature_Extractor(Exp[i]),Weights)+1)
        Theta=np.reshape(Theta,[3,4])
        for k in range (100):
            aaa=Action_Value.Least_Square_Q_Learning(Exp[np.max([0,i-400]):i+1],Theta)
            Weights+=aaa
            if i==49:
                W_T[k,:]=aaa
        Weights/=100
def Evaluate_Action_Value_Functionn(Weights,Experiences):
    
    Num_Exp=np.shape(Experiences)[0]
    Error=np.zeros(Num_Exp)
    
    for i in range(Num_Exp-1):
        Feature_t=np.concatenate([[1],Experiences[i][0],Experiences[i][1]])
#        Feature_t[-1]=np.log(Feature_t[-1])
        Feature_t_Plus_1=np.concatenate([[1],Experiences[i][3],Experiences[i][1]])
#        Feature_t_Plus_1[-1]=np.log(Feature_t_Plus_1[-1])
        
        
        
        Predicted_Reward=Action_Value.Linear_Action_Value(Feature_t,Weights)-Action_Value.Linear_Action_Value(Feature_t_Plus_1,Weights)
        
        Error[i]=(Experiences[i][2]-Predicted_Reward)/Experiences[i][2]
    return Error

plt.hist(Evaluate_Action_Value_Functionn(Weights,Exp[50:250]),100)
print(np.mean(np.abs(Evaluate_Action_Value_Functionn(Weights,Exp[50:250  ]))))
#imp.rel


#oad(Action_Value)
#Weights=Action_Value.Least_Square_Q_Learning(Exp,Theta)
#Weights=np.zeros(8)
#for k in range (2):
#    Weights+=Action_Value.Least_Square_Q_Learning(Exp[0:300],Theta)
#Weights/=2
#print(Weights)



#plt.hist(ab1[:,5],500)
#ab1=W_T
#ab2=Weights

print(Weights)
#print(np.mean(ab1,axis=0))
#print(np.std(W_T,axis=0))
# 
#print(Action_Value.Q(Exp[0][0],Exp[2][1],Weights))
#print(Action_Value.Q(Exp[0][0],Exp[3][1],Weights))
#
#
#print(Exp[6])

























