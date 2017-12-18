import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance
import csv
import Action_Value
import imp
import time
imp.reload(Action_Value)
a1=[0,np.cos(np.pi/6),np.cos(np.pi/3),60]
a2=[np.cos(np.pi/6),0,np.cos(np.pi/3),60]
a3=[-np.cos(np.pi/6),0,np.cos(np.pi/3),60]
a4=[0,-np.cos(np.pi/6),np.cos(np.pi/3),60]
a34=[-np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3),60]
a24=[np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3),60]
##
#
Exp_x39_z3=Action_Value.Create_Batch_Experience(X=0.1,Z=4)
np.save('Exp_x01_z40_ComfortOnly.npy',Exp_x39_z3)
#
#Exp=np.load('Exp2.npy').tolist()
#
#
#
#a=np.array([1,2,3]).astype(float)
#a=np.insert(a,[0],6.3)

#print(a)
#print(Action_Value.Data_Reader(a1,X=3.9,Z=3.2))
#print(Action_Value.Data_Reader(a2,X=3.9,Z=3.2))
#print(Action_Value.Data_Reader(a3,X=3.9,Z=3.2))
#print(Action_Value.Data_Reader(a4,X=3.9,Z=3.2))
#print(Action_Value.Data_Reader(a34,X=3.9,Z=3.2))
#print(np.shape([0,0,0,0])[0])
#a=Action_Value.Gaussian_Policy_Score_Function(Action_Value.Create_Experience(a24),np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
#print(np.shape(a24)[0])
##print(a)
#a=Action_Value.Create_Batch_Experience()
#Exp=Action_Value.Create_Batch_Experience()
#with open("file.txt", "w") as output:
#    output.write(str(Exp))









#W=np.zeros(8)
#Elig=W
#delta_W=0
####for i in range(np.shape(Exp)[0]):
####    if Exp[i][2]>=0:
####        print('--------------------------------------------')
####        print(Exp[i][2])
#Err=np.zeros(100)
#
#for j in range(1000):
#    
##    for i in range(np.shape(Exp)[0]):
#    for i in range(100):
#        
#        delta_W,Elig,Error=Action_Value.Linear_Action_Value_Function_Weight_Updator_TD_Lambda(W,Exp[i],Elig,0)
#        
#        W+=delta_W
##        print(delta_W)
##        print(np.sum(np.abs(delta_W/W)),Exp[i][1][3])
##        print('------------------------------------------------------')
#        
#        Err[i]=Error/Exp[i][2]
#        print(np.sum(np.abs(W)))
#plt.hist(Err,300)
#print(np.sum(np.abs(Err)<0.3))
#





a=np.linspace(0,2*np.pi,1000)

X=np.cos(a)
Y=np.sin(a)
Synth_Features=np.array([X,Y,0.5*np.ones(1000),0.725*np.ones(1000),X,Y,0.5*np.ones(1000),10*np.ones(1000)]).T


##
#for i in range(np.shape(Exp)[0]):
#    
#    Exp[i][1][3]=Exp[i][3][3]-Exp[i][0][3]
#    Exp[i][0][3]=0
#    Exp[i][3][3]=0
#print(Exp[i][1][0:3])





#c=b.tolist()
