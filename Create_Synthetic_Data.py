import numpy as np
import sklearn
import imp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression             as LR
from sklearn.neural_network import MLPRegressor               as NNR
from sklearn.tree import DecisionTreeRegressor                as DTR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

#####################################################################################################
############################## Function Definition Part #############################################

def Direction_Isolater(Experiences):
    Result=dict()
#    Directions=np.unique([Exp[i][0][0:3] for i in range(len(Exp))],axis=0)
    Directions_Dict=dict()
    d1=np.array([0,np.cos(np.pi/6),np.cos(np.pi/3)])
    d2=np.array([np.cos(np.pi/6),0,np.cos(np.pi/3)])
    d3=np.array([-np.cos(np.pi/6),0,np.cos(np.pi/3)])
    d4=np.array([0,-np.cos(np.pi/6),np.cos(np.pi/3)])
    d24=np.array([np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3)])
    d34=np.array([-np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3)])
    Directions=[d1,d2,d3,d4,d24,d34]
    for i,key in enumerate(Directions):
        Directions_Dict[i]=key
        Result[i]=[]
    
    for i,item in enumerate(Experiences):
        for j in range(6):
            
            if np.mean(item[0][0:3]==Directions_Dict[j])==1:
                Result[j].append(item)
                break
   
    return Result,Directions_Dict
    
def Build_Experience_By_Direction(Experiences,Directions,Random=False):
    n=len(Directions)
    Temp0,Temp1=Direction_Isolater(Experiences)
    Result=[]
    for i,j in enumerate(Directions):
        Result.append(Temp0[j])
    Result=np.concatenate(Result)
    if Random==True:
        np.random.shuffle(Result)
        
    Result=np.ndarray.tolist(Result)
    
    return Result
    
    
def Feature_Extracor(Experiences):
    Features=[]
    Rewards=[]
    for i in range(np.shape(Experiences)[0]):
        Feature_t=np.concatenate([Experiences[i][0],Experiences[i][1]])
        Features.append(Feature_t)
        Rewards.append(Experiences[i][2])
    X=np.array(Features)
    Y=np.array(Rewards)  

    Y=Y.ravel()
    return X,Y

def Regression_Method_Evaluator(Dir_Include,Dir_Evaluate,Method,plot=False):
    'Load Experience'
    Exp=np.load('Exp.npy').tolist()
    
    Training_Exp=Build_Experience_By_Direction(Exp,Dir_Include,Random=True)
    Testing_Exp=Build_Experience_By_Direction(Exp,Dir_Evaluate,Random=True)
    Training_Data,Training_Labels=Feature_Extracor(Training_Exp)
    Testing_Data,Testing_Labels=Feature_Extracor(Testing_Exp)
    
    Model=Method
    Model.fit(Training_Data,Training_Labels)
    Prediction_Test=Model.predict(Testing_Data)
    Prediction_Train=Model.predict(Training_Data)
    Mean_Testing_Abs_Error=np.mean(np.abs((Prediction_Test-Testing_Labels)/Testing_Labels)*100)
    Mean_Training_Abs_Error=np.mean(np.abs((Prediction_Train-Training_Labels)/Training_Labels)*100)
    print('--------------------------------------------------')
    print('Mean absolute test_data error:',Mean_Testing_Abs_Error)
    print('Mean absolute train_data error:',Mean_Training_Abs_Error)
    print('--------------------------------------------------')
    
    if plot==True:
        plt.hist(np.abs((Prediction_Test-Testing_Labels)/Testing_Labels)*100,300)
 
#####################################################################################################




Regression_Method_Evaluator([2,3],[5],GPR(normalize_y=True), plot=True)































