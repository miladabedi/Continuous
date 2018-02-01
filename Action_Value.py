 6gimport numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance
import csv
import copy
#def Linear_Action_Value_Function_Weight_Updator(Initial_Weight_Vector,Experience,Parameters):
#    """
#    Experience=np.array([[a,b,c,maybe_price],[a',b',c',maybe_price'],[reward],[a',b',c',maybe_price']])
#        
#    
#    """
#    Alpha=0.001
#    
#    
#    Target="............................. Write a Subroutine that would take the method and calculate the target accordingly"    
##    Feature_Vector=np.concatenate((Experience[0],Experience[1])) " We can explore the Features later. For now assume features consist of x,y,z,price "
#    Feature_Vector=Experience[1] "feature vector is in form of X(S,A), but since actions as defined fully represent the next state and keeping in mind the indipendence of q(S,A) from the current"
#                                 "state the features of the action itself fully represent the q(S,A)"
#    
#    
#    Delta_W=Alpha*(Target-Linear_Action_Value(Feature_Vector,Initial_Weight_Vector))*Feature_Vector
#    New_Weight=Initial_Weight_Vector+Delta_W
#    return New_Weight


#def RL_Agent_Trainer():
    
    
    
    
    
    
    
    
    
def Create_Batch_Experience(X=2,Z=4):
    Time_Steps=np.linspace(10,420,42)
    Exp_Batch=[]
    Directions=np.array([[0,np.cos(np.pi/6),np.cos(np.pi/3)],[np.cos(np.pi/6),0,np.cos(np.pi/3)],[-np.cos(np.pi/6),0,np.cos(np.pi/3)],[0,-np.cos(np.pi/6),np.cos(np.pi/3)],[-np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3)],[np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3)]])
    for i in Time_Steps:
        count=i
        while count<=420:
            for j in Directions:
                PC=Data_Reader(np.concatenate((j,[count-i])),X,Z)
                Exp_Batch.append(Create_Single_Experience(np.concatenate((j,[i])),np.concatenate((j,[PC])),X,Z))
            count+=i
    
    return Exp_Batch
def Create_Single_Experience(Action,Original_State=np.array([0,0,0,0]),X=3,Z=1):
    PC=Data_Reader(Action,X,Z)
    Experience=np.array([Original_State,np.array(Action),np.array([Reward_Function(PC,Action[3])-Original_State[3]]),np.append(Action[0:3],PC)])
    return Experience

def Least_Square_Q_Learning(Experiences,Theta,Discount_Factor=1):

    A=np.zeros([9,9])
    B=np.zeros(9)
    for i in range(np.shape(Experiences)[0]):
        Next_Action=Gaussian_Policy_Sampler(Experiences[i][0],Theta)
        Feature_t=copy.deepcopy(np.concatenate([[1],Experiences[i][0],Experiences[i][1]]))
#        Feature_t[-1]=np.log(Feature_t[-1])
        Feature_t_plus_1=copy.deepcopy(np.concatenate([[1],Experiences[i][3],Next_Action]))
##        Feature_t_plus_1[-1]=np.log(Feature_t_plus_1[-1])

        A+=np.outer(Feature_t,Feature_t-Discount_Factor*Feature_t_plus_1)
        B+=Feature_t*Experiences[i][2]
    
    Weights=np.matmul(np.linalg.pinv(A),B)
    return Weights    
    


def Linear_Action_Value_Function_Weight_Updator_TD_Lambda(Initial_Weight_Vector,Experience,Eligibility_Vector,Lambda,Alpha=0.2,Discount_Factor=0.99):
    #Feature_Vector_State_t=np.concatenate((Experience[0],Experience[1]))

    
    Feature_Vector_State_t=copy.deepcopy(Experience[0])
    Feature_Vector_Action_t_plus_1=copy.deepcopy(Experience[1])
#    Feature_Vector_Action_t_plus_1[3]=np.log(Feature_Vector_Action_t_plus_1[3])
    Feature_Vector_State_t_plus_1=copy.deepcopy(Experience[3])
    Feature_Vector_t=np.concatenate([Feature_Vector_State_t,Feature_Vector_Action_t_plus_1])
    Feature_Vector_t_plus_1=np.concatenate([Feature_Vector_State_t_plus_1,Feature_Vector_Action_t_plus_1])
    
    Reward=Experience[2]
    TD_Error=Reward+Discount_Factor*(Linear_Action_Value(Feature_Vector_t_plus_1,Initial_Weight_Vector))-Linear_Action_Value(Feature_Vector_t,Initial_Weight_Vector)
    Eligibility_Vector=Discount_Factor*Lambda*Eligibility_Vector+Feature_Vector_t
    Delta_W=Alpha*TD_Error*Eligibility_Vector
    
    return Delta_W,Eligibility_Vector,TD_Error
    

def Gaussian_Policy_Sampler(State,Policy_Parameters,Sigma=0.35):
    State_Features=copy.deepcopy(State)
    State_Features[0:3]/=State_Features[2]
    State_Features=np.delete(State_Features,2)
    State_Features=np.insert(State_Features,[0],1) #Intercept
    Sample=np.array([np.random.normal(np.inner(State_Features,Policy_Parameters[j]),Sigma,1) for j in range(3)]).T[0]
    Sample=np.insert(Sample,2,1)
    
    Sample[0:3]/=np.linalg.norm(Sample[0:3])
    Sample[3]=np.abs(Sample[3])
    
    return(Sample)

def Gaussian_Policy_Score_Function(Experience,Current_Theta,Params={'Sigma':.35}):
    
    Sigma=Params['Sigma']
    Current_Theta=np.reshape(Current_Theta,[3,4])
    """
    
    Current_Theta is a array of size n*m where n is determined by the size of action space and m is equal to the size of feaures.
    each column within Current_Theta represents the parameters of the normal distribution function over the corresponding action in the action space
    For the purposes of mathematics, Current_Theta should be imagined as a vector of concatenated thetas
    """
    State=copy.deepcopy(Experience[0])
    Action=copy.deepcopy(Experience[1])
    Action[0:3]/=Action[2]
#    Action[-1]=np.log(Action[-1])
    Action=np.delete(Action,2)
#    State_Features=Feature_Extractor_For_Policy_Function(State) "Feature Extraction To be completed"
    State_Features=State
    State_Features[0:3]/=State_Features[2]
    State_Features=np.delete(State_Features,2)
    State_Features=np.insert(State_Features,[0],1) #Intercept
    Action_Space_Size=3
    
    Parameter_Vector_Size=np.shape(Current_Theta[0])[0]
    Score_Value=np.array([np.zeros(Parameter_Vector_Size) for i in range(Action_Space_Size)])
    
    
    for i in range(Action_Space_Size):
        
        Score_Value[i]=(Action[i]-np.inner(State_Features,Current_Theta[i]))*State_Features/Sigma**2
#        print(Score_Value[i])
    Score_Value=Score_Value.ravel()
    return Score_Value




def Single_Experience_Policy_Parameter_Adjustment(Policy_Score_Function,Experience,Current_Theta,Step_Size=0.1,Params={}):
    
    """
    
    Current_Theta is a array of size n*m where n is determined by the size of action space and m is equal to the size of feaures.
    each column within Current_Theta represents the parameters of the normal distribution function over the corresponding action in the action space
    For the purposes of mathematics, Current_Theta should be imagined as a vector of concatenated thetas
    
    
    
    The Same goes for Delta_Theta
    Sinse both Current_Theta and Delta_Theta have been defined as numpy arrays (and naturally both are the same size) We dont have to loop over 
    individual elements and we can treat the whole thing as a matrix as far as basic operations are concerned
    """
    
    Reward=Experience[2]
    Delta_Theta=Step_Size*Policy_Score_Function(Experience,Current_Theta,Params)*Reward
    return Delta_Theta
      
    
    
def Linear_Action_Value(Features,Weight_Vector):
    Action_Value=np.inner(Features,Weight_Vector)
    return Action_Value



def S_A_Feature_Extractor(Experience):
    Feature=copy.deepcopy(np.concatenate([[1],Experience[0],Experience[1]]))
#    Feature[-1]=np.log(Feature[-1])

    return Feature

        


def Most_Likely_Action_Gaussian_Action_Probability(State,Current_Theta):
    n,m=Current_Theta.shape
    Most_Likely_Action=np.array([np.inner(Feature_Extractor_For_Policy_Function(State),Current_Theta[i]) for i in range(n)])
    return Most_Likely_Action


def Optimum_Feature(Weight_Vector,Function=Linear_Action_Value,theta_max=np.pi/3):
    
    "If I decide to change the feature space (from [x,y,z]) I should rewrite this function from scratch"
    
    
    
    
    Cons=({'type':'ineq',
           'fun':lambda x: theta_max-np.inner(x,np.array([0,0,1]))/np.linalg.norm(x),
           'jac':lambda x: -(np.array([0,0,1])/np.linalg.norm(x)-x*np.inner(x,np.array([0,0,1]))/(np.linalg.norm(x)**3))
           })
    res=minimize(lambda F:Function(F,Weight_Vector), jac=lambda F:Weight_Vector,x0=np.array([0,0,1]),constraints=Cons)
    Opt_Feat=res.x
    return Opt_Feat



def Data_Reader(action,X=3,Z=1):
    
    d1=str(np.round(([0,np.cos(np.pi/6),np.cos(np.pi/3)]),5))
    d2=str(np.round(([np.cos(np.pi/6),0,np.cos(np.pi/3)]),5))
    d3=str(np.round(([-np.cos(np.pi/6),0,np.cos(np.pi/3)]),5))
    d4=str(np.round(([0,-np.cos(np.pi/6),np.cos(np.pi/3)]),5))
    d24=str(np.round([np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3)],5))
    d34=str(np.round([-np.cos(np.pi/6)*np.cos(np.pi/4),-np.cos(np.pi/6)*np.cos(np.pi/4),np.cos(np.pi/3)],5))
    Folder_Name={d1:'uvwT(1)',d2:'uvwT(2)',d3:'uvwT(3)',d4:'uvwT(4)',d24:'uvwT(24)',d34:'uvwT(34)'}
    Direction=str(np.round(action[:3],5))
    Duration=str(int((action[3]*10+1800)))
    path= "E:/Virginia Tech/Research/HVAC/4-Direction CFD/Data/"+Folder_Name[Direction]+"/"+Duration
    file=open(path,newline='')
    reader=csv.reader(file)
    header=next(reader)
    data=[row for row in reader]
    i=0
    Data=np.matrix(np.zeros((len(data),len(data[0]))))
    for item in data:
        Data[i,:]=list(map(float,item))
        i+=1
    Read_Locations=[[X,0,Z],[X,0.6,Z],[X,1.2,Z]]
    nodes=[Nearest_Node(Data,Read_Locations[0]),Nearest_Node(Data,Read_Locations[1]),Nearest_Node(Data,Read_Locations[2])]
    Temperature=[Data[nodes[0],7],Data[nodes[1],7],Data[nodes[2],7]]
    
    
    Prop_Comfort=Temperature_to_Proportion_Comfortable(np.average(Temperature))
    
    
    
    
    
    
    
    return Prop_Comfort

def Q(State,Action,Weights):
    Feature=S_A_Feature_Extractor(np.array([State,Action]))
    result=Linear_Action_Value(Feature,Weights)
    return result
def Nearest_Node(Data,Location):
    Dist_Sqr=np.zeros(len(Data))
    for i in range(len(Data)):
        Dist_Sqr[i]=(Data[i,1]-Location[0])**2+(Data[i,2]-Location[1])**2+(Data[i,3]-Location[2])**2
    node=np.argmin(Dist_Sqr)
    return node


def Temperature_to_Proportion_Comfortable(Temperature):
    Graph=np.matrix([[15,0.57/0.81],[20,0.79/0.81],[22.5,0.81/0.81],[23.75,0.76/0.81],[25,0.68/0.81],[27.5,0.39/0.81],[30,0.14/0.81],[32.5,0.04/0.81],[35,0/0.81]])
    Temperature=Temperature-273.15
    x2=np.argmax(Graph[:,0]>=Temperature)
    x1=x2-1
    y2=Graph[x2,1]
    y1=Graph[x1,1]
    T1=Graph[x1,0]
    T2=Graph[x2,0]
    PC=-(y2-y1)/(T2-T1)*(T2-Temperature)+y2
    return PC


def Reward_Function(Prop_Comfort,Duration,max_duration=420):
    return -Duration/max_duration*0+Prop_Comfort


