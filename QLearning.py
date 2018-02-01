import numpy as np
import sklearn
import imp
from pandas import DataFrame
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression             as LR
from sklearn.neural_network import MLPRegressor               as NNR
from sklearn.tree import DecisionTreeRegressor                as DTR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import copy
from scipy import optimize,stats
#####################################################################################################
############################## Function Definition Part #############################################
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
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
    
    
def Feature_Extractor(Experiences):
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

def Q_Function(Training_List,Method,Exp_File_Name,Updated_Training_Labels=[],plot=False,Duration_Limit=[],Return_Model=False):
    'Load Experience'
    
    Experiences=np.load(Exp_File_Name).tolist()
    global initial_comfort,Prediction_Test,Testing_Labels,Testing_Data,Training_Data,Prediction_Train, Training_Labels,Testing_List,Mean_Testing_Abs_Error
    initial_comfort=Experiences[0][0][3]
    if np.any(Duration_Limit):
        Experiences=Duration_Limiter(Experiences,Duration_Limit)
    
    Dir_Include=[0,1,2,3,4,5]
    
    
    
    
    
    
    
    
    All_Exp=Build_Experience_By_Direction(Experiences,Dir_Include,Random=False)
#    Testing_Exp=Build_Experience_By_Direction(Experiences,Dir_Evaluate,Random=True)
    All_Data,All_Labels=Feature_Extractor(All_Exp)
    Testing_List=np.delete(np.arange(0,np.shape(All_Data)[0]),Training_List)
    

#    Testing_Data,Testing_Labels=Feature_Extractor(Testing_Exp)
    Training_Data=All_Data[Training_List,:]
    Testing_Data=All_Data[Testing_List,:]
    Testing_Labels=All_Labels[Testing_List]
    if np.any(Updated_Training_Labels):
        Training_Labels=Updated_Training_Labels
    else:
        Training_Labels=All_Labels[Training_List]


    
    Model=Method
    Model.fit(Training_Data,Training_Labels)
    Prediction_Test=Model.predict(Testing_Data)
    Prediction_Train=Model.predict(Training_Data)
    
    Testing_Error_Array=np.array((Prediction_Test-Testing_Labels))
    Training_Error_Array=np.array((Prediction_Train-Training_Labels))
    Mean_Testing_Abs_Error=np.mean(np.abs(Testing_Error_Array))
    Mean_Training_Abs_Error=np.mean(np.abs(Training_Error_Array))
#    print('--------------------------------------------------')
#    print('Mean absolute test_data error:',Mean_Testing_Abs_Error)
#    print('--------------------------------------------------')
    
    if plot==True:
        plt.hist(Testing_Error_Array,300)
        plt.show()
        
    if Return_Model==True:
        return Model

def Duration_Limiter(All_Experiences,Duration_Limit):
    
    
    Iso_Exp,Directions=Direction_Isolater(All_Experiences)
    keys=list(Iso_Exp.keys())
    Result=[]
    
    for i in keys:
        

        Start=0
        Duration_Count=0
        Condition_1=True
        
        for j in range(len(Iso_Exp[i])):
    
            if Condition_1==False:
                if np.round(Iso_Exp[i][j][1][3],4)!=np.round(Iso_Exp[i][j-1][1][3],4) and Iso_Exp[i][j][1][3]<=Duration_Limit:        
                    Condition_1=True
                    Start=j
                    
            if Condition_1 :
                Duration_Count+=Iso_Exp[i][j][1][3]
                
                
                if np.round(Duration_Count+Iso_Exp[i][j][1][3],2)>Duration_Limit:
    
                    Result.extend(Iso_Exp[i][Start:j+1])
                    Duration_Count=0
                    Condition_1=False
    
                    
    return Result

def normalize(Input,transform=[]):

    Result=(Input-Input.mean())/(Input.max()-Input.min())
    return Result

def Target_Function(Input,Model):
    Angle=Input[0]
    Duration=Input[1]
    X=np.sqrt(0.75)*np.cos(Angle)
    Y=np.sqrt(0.75)*np.sin(Angle)
    Synth_Features=np.array([X,Y,0.5,initial_comfort,X,Y,0.5,Duration]).reshape(1,-1)
    return Model.predict(Synth_Features)

def Plot_Direction_Reward(Model,Coord_x,Coord_z):
    n=360
    a=np.linspace(0,2*np.pi,n)
    X=np.sqrt(0.75)*np.cos(a)
    Y=np.sqrt(0.75)*np.sin(a)

    for i in np.arange(60,70,10):
        Synth_Features=np.array([X,Y,0.5*np.ones(n),initial_comfort*np.ones(n),X,Y,0.5*np.ones(n),i*np.ones(n)]).T
        x=-Y*50+150
        y=-X*50+300
        z = Model.predict(Synth_Features)
   
        
    
        cmap = plt.get_cmap('jet', 20)
        cmap.set_under('gray')
        
        fig, ax = plt.subplots()
        fig.dpi=150
        ax.set_aspect('equal')
        cax = ax.scatter(x, y, c=z, s=3, cmap=cmap, vmin=z.min(), vmax=z.max())
        ax.scatter(-Coord_z*100+683,Coord_x*100+102,c='r',s=5)
        plt.title(i)
        fig.colorbar(cax, extend='min')
        im=plt.imread('Room.png')
        plt.imshow(im)    
        plt.show()
def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.cross(v1, v2)
    result=np.arctan2(sinang, cosang)
    if result<0:
        result+=2*np.pi
    
    return result

def RelativeAngle_DurationRequired(Coord_x,Coord_z,Model,Min_Comfort_Level=0.95,Plot=False):
    global User_Direction
    Relative_x=Coord_x-2
    Relative_z=Coord_z-5.4
    Directions=np.linspace(0,359,360)/360*np.pi*2
    Durations=np.ones(np.shape(Directions))*-1
    for i in enumerate(Directions):
        for j in np.arange(10,430,10):
            if Target_Function([i[1],j],Model)>=Min_Comfort_Level-initial_comfort:
                Durations[i[0]]=j
                break
            
    User_Direction=np.round(py_ang(np.array([1,0]),np.array([-Relative_x,Relative_z]))/np.pi*180)
    Directions=np.linspace(0,359,360)-User_Direction
    for i in enumerate(Directions):
        if i[1]>180:
            Directions[i[0]]-=360
        if i[1]<-180:
            Directions[i[0]]+=360
    Result=np.array([Directions,Durations]).T
    Result=Result[np.argsort(Result[:,0],0),:]
    if Plot==True:
        plt.plot(Result[:,0],Result[:,1])
#        plt.show()
    return Result
def Find_Max_Q(Next_State,Model):
    Max_Q=-1000
    Action=copy.deepcopy(Next_State)
    for i in range(0,430,10):
        Action[-1]=i
        State_Action_Pair=np.concatenate((Next_State,Action)).reshape(1,-1)
#        print(State_Action_Pair)
        if Model.predict(State_Action_Pair)>Max_Q:
            Max_Q=np.max([Max_Q,Model.predict(State_Action_Pair)])
            Best_Duration=i
        
    return Max_Q,Best_Duration
####################################################################################################################################################################   

list1=np.random.choice(1008,size=50,replace=False)
Model=Q_Function(list1,GPR(normalize_y=True),Exp_File_Name='Exp_x20_z10_ComfortOnly.npy', plot=False,Return_Model=True)
rewards=copy.deepcopy(Training_Labels)
Updated_Training_Labels=np.zeros(np.shape(rewards))
a=0
for k in range(1000):
    count=0
    for i in list1:
        Next_State=Training_Data[count,0:4]
        Updated_Training_Labels[count]=rewards[count]+.95*Find_Max_Q(Next_State,Model)[0]
        count+=1
    Model=Q_Function(list1,GPR(normalize_y=True),Updated_Training_Labels=Updated_Training_Labels,Exp_File_Name='Exp_x20_z10_ComfortOnly.npy', plot=False,Return_Model=True)
    print(np.sum(Updated_Training_Labels)-a,k)
    a=np.sum(Updated_Training_Labels)



a=Updated_Training_Labels-Updated_Training_Labels[0]+rewards[0]

plt.scatter(rewards,a)


