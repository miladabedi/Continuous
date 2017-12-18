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

def Regression_Method_Evaluator(Dir_Include,Dir_Evaluate,Method,plot=False,Duration_Limit=[],Return_Model=False):
    'Load Experience'
    global Testing_Data
    Experiences=np.load('Exp_x01_z40.npy').tolist()
    
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
    

    
    Mean_Testing_Abs_Error=np.mean(np.abs((Prediction_Test-Testing_Labels)/Prediction_Test)*100)
    Mean_Training_Abs_Error=np.mean(np.abs((Prediction_Train-Training_Labels)/Prediction_Train)*100)
    print('--------------------------------------------------')
    print('Mean absolute test_data error:',Mean_Testing_Abs_Error)
    print('Mean absolute train_data error:',Mean_Training_Abs_Error)
    print('--------------------------------------------------')
    
    if plot==True:
        plt.hist(np.abs((Prediction_Test-Testing_Labels)/Prediction_Test)*100,300)
        
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
def update_plot(i,data,scat):
    scat.set_array(data[i])
    return scat,

#####################################################################################################



Model=Regression_Method_Evaluator([0,1,2,3,4,5],[1],GPR(normalize_y=True), plot=False,Return_Model=True)
#Regression_Method_Evaluator([2],[5],LR(), plot=True,Duration_Limit=20)
a=np.linspace(0,2*np.pi,1000)
X=np.sqrt(0.75)*np.cos(a)
Y=np.sqrt(0.75)*np.sin(a)


#for i in np.arange(10,400,10):
#    Synth_Features=np.array([X,Y,0.5*np.ones(1000),0.725*np.ones(1000),X,Y,0.5*np.ones(1000),i*np.ones(1000)]).T
#    
#    r=np.array([Model.predict(Synth_Features),X,Y]).T
#    color=r[:,0]
#    plt.axes().set_aspect('equal', 'datalim')
#    plt.scatter(X,Y,c=color)
#    plt.scatter(-1.9,-1.4,c='r')
#    plt.title(i)
#    plt.show()
#
#    time.sleep(0.1)




for i in np.arange(10,400,10):
    Synth_Features=np.array([X,Y,0.5*np.ones(1000),0.725*np.ones(1000),X,Y,0.5*np.ones(1000),i*np.ones(1000)]).T
    x=X
    y=Y
    # Generate some data
    z = Model.predict(Synth_Features)
    # Set some values in z to 0...
    #z[:5] = 0
    
    cmap = plt.get_cmap('jet', 20)
    cmap.set_under('gray')
    
    fig, ax = plt.subplots()
    cax = ax.scatter(x, y, c=z, s=100, cmap=cmap, vmin=z.min(), vmax=z.max())
    ax.scatter(-1.9,-1.4,c='r')
    plt.title(i)
    fig.colorbar(cax, extend='min')
    
    plt.show()






