import numpy as np
import sklearn
import imp
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression             as LR
from sklearn.neural_network import MLPRegressor               as NNR
from sklearn.tree import DecisionTreeRegressor                as DTR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import time
from scipy import optimize,stats
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

def Regression_Method_Evaluator(Dir_Include,Dir_Evaluate,Method,Exp_File_Name,plot=False,Duration_Limit=[],Return_Model=False):
    'Load Experience'
    Experiences=np.load(Exp_File_Name).tolist()
    global initial_comfort,Testing_Error_Array,Training_Error_Array
    initial_comfort=Experiences[0][0][3]
    if np.any(Duration_Limit):
        Experiences=Duration_Limiter(Experiences,Duration_Limit)
    
    
    
    
    
    
    
    
    
    
    
    Training_Exp=Build_Experience_By_Direction(Experiences,Dir_Include,Random=True)
    Testing_Exp=Build_Experience_By_Direction(Experiences,Dir_Evaluate,Random=True)
    Training_Data,Training_Labels=Feature_Extractor(Training_Exp)
    Testing_Data,Testing_Labels=Feature_Extractor(Testing_Exp)
    
    Model=Method
    Model.fit(Training_Data,Training_Labels)
    Prediction_Test=Model.predict(Testing_Data)
    Prediction_Train=Model.predict(Training_Data)
    
    Testing_Error_Array=np.array((Prediction_Test-Testing_Labels))
    Training_Error_Array=np.array((Prediction_Train-Training_Labels))
    Mean_Testing_Abs_Error=np.mean(np.abs(Testing_Error_Array))
    Mean_Training_Abs_Error=np.mean(np.abs(Training_Error_Array))
    print('--------------------------------------------------')
    print('Mean absolute test_data error:',Mean_Testing_Abs_Error)
    print('--------------------------------------------------')
    
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
    a=np.linspace(0,2*np.pi,1000)
    X=np.sqrt(0.75)*np.cos(a)
    Y=np.sqrt(0.75)*np.sin(a)
    for i in np.arange(10,430,10):
        Synth_Features=np.array([X,Y,0.5*np.ones(1000),initial_comfort*np.ones(1000),X,Y,0.5*np.ones(1000),i*np.ones(1000)]).T
        x=X
        y=Y
        z = Model.predict(Synth_Features)
    
        cmap = plt.get_cmap('jet', 20)
        cmap.set_under('gray')
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        cax = ax.scatter(x, y, c=z, s=100, cmap=cmap, vmin=z.min(), vmax=z.max())
        ax.scatter(Coord_x-2,-5.4+Coord_z,c='r')
        plt.title(i)
        fig.colorbar(cax, extend='min')
    
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
    
    Relative_x=Coord_x-2
    Relative_z=Coord_z-5.4
    Directions=np.linspace(0,359,360)/360*np.pi*2
    Durations=np.ones(np.shape(Directions))*-1
    for i in enumerate(Directions):
        for j in np.arange(10,430,10):
            if Target_Function([i[1],j],Model)>=Min_Comfort_Level-initial_comfort:
                Durations[i[0]]=j
                break
            
    User_Direction=np.round(py_ang(np.array([1,0]),np.array([Relative_x,Relative_z]))/np.pi*180)
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
        plt.show()
    return Result

#####################################################################################################



#Model=Regression_Method_Evaluator([0,1,2,3,4,5],[1],GPR(normalize_y=True),Exp_File_Name='Exp_x25_z30_ComfortOnly.npy', plot=False,Return_Model=True)
#Regression_Method_Evaluator([2],[5],LR(), plot=True,Duration_Limit=20)
#Plot_Direction_Reward(Model,2.5,3)
#optimize.brute(Target_Function,(slice(0,2*np.pi,2*np.pi/360),slice(0,420,10)))


x=[5,10,15,20,25,30,35]

z=[5,10,15,20,25,30,35,40,45,50,55]
z=[5]
for j in z:
    for i in x:
        filename='Exp_x'+str(i)+'_z'+str(j)+'_ComfortOnly.npy'
        Model=Regression_Method_Evaluator([0,1,2,3],[4],GPR(normalize_y=True),Exp_File_Name=filename, plot=False,Return_Model=True)
        print('x=',i/10,'z=',j/10)
        a=RelativeAngle_DurationRequired(i/10,j/10,Model,Min_Comfort_Level=0.95,Plot=True)
#    print(j)
#    plt.show()
#        plt.plot(a,Prediction_Test)
#        plt.plot(a,Testing_Labels)
#        plt.show()
