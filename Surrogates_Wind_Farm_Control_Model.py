import chaospy as cp
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random

########## Loading the necessary training data to develop the surrogate model#################################################################
##############################################################################################################################################

file_name = 'INSERT_PATH_NAME'

## Loading the power of the upstream turbine
Input_Power1 = scipy.io.loadmat(file_name+'Input_Run_Power_Turb1.mat')
Input_Power1 = Input_Power1['Input_Run_Power_Turb1']

## Loading the power of the downstream turbine
Input_Power2 = scipy.io.loadmat(file_name+'Input_Run_Power_Turb2.mat')
Input_Power2 = Input_Power2['Input_Run_Power_Turb2']

#Scaling the input paramters of the upstream turbine
yaw_1_train_scaled = Input_Power1[0]/30
Power_scaled = Input_Power1[1]/(np.max(Input_Power1[1]) + np.max(Input_Power2[3]))

Input_Power1 = None
Input_Power1 = [yaw_1_train_scaled, Power_scaled]


# Selecting the desired training domain for the downstream turbine
I1 = Input_Power2[2]>3
I2 = Input_Power2[2]<19

I3 = Input_Power2[2][I1*I2]==10

#Scaling the input paramters of the downstream turbine
yaw_1_train_scaled = Input_Power2[0][I1*I2]/30
yaw_2_train_scaled = Input_Power2[1][I1*I2]/30
Distance_train_scaled = Input_Power2[2][I1*I2]/np.max(Input_Power2[2])
Power_scaled = Input_Power2[3][I1*I2]/(np.max(Input_Power1[1]) + np.max(Input_Power2[3]))

Input_Power2 = None
Input_Power2 = [yaw_1_train_scaled, yaw_2_train_scaled, Distance_train_scaled, Power_scaled]

## Loading the Flapwise Bending Moment of the upstream turbine 
Input_FlapM_Turb1 = scipy.io.loadmat(file_name+'Input_Run_FlapM_Turb1.mat')
Input_FlapM_Turb1 = Input_FlapM_Turb1['Input_Run_FlapM_Turb1']

yaw_1_train_scaled = Input_FlapM_Turb1[0]/30
FlapM_scaled = Input_FlapM_Turb1[1]/np.max(Input_FlapM_Turb1[1])

Input_FlapM_Turb1 = None
Input_FlapM_Turb1 = [yaw_1_train_scaled, FlapM_scaled]

## Loading the Flapwise Bending Moment of the downstream turbine 
Input_FlapM_Turb2 = scipy.io.loadmat(file_name+'Input_Run_FlapM_Turb2.mat')
Input_FlapM_Turb2 = Input_FlapM_Turb2['Input_Run_FlapM_Turb2']

yaw_1_train_scaled = Input_FlapM_Turb2[0][I1*I2]/30
yaw_2_train_scaled = Input_FlapM_Turb2[1][I1*I2]/30
Distance_train_scaled = Input_FlapM_Turb2[2][I1*I2]/np.max(Input_FlapM_Turb2[2])
FlapM_scaled = Input_FlapM_Turb2[3][I1*I2]/np.max(Input_FlapM_Turb2[3])

Input_FlapM_Turb2 = None
Input_FlapM_Turb2 = [yaw_1_train_scaled, yaw_2_train_scaled, Distance_train_scaled, FlapM_scaled]

## Loading the combined Tower Bending Moment of the upstream turbine 
Input_MZ_MY_Turb1 = scipy.io.loadmat(file_name+'Input_Run_MZ_MY_Turb1.mat')
Input_MZ_MY_Turb1 = Input_MZ_MY_Turb1['Input_Run_Mz_My_Turb1']

yaw_1_train_scaled = Input_MZ_MY_Turb1[0]/30
MZ_MY_scaled = Input_MZ_MY_Turb1[1]/np.max(Input_MZ_MY_Turb1[1])

Input_MZ_MY = None
Input_MZ_MY_Turb1 = [yaw_1_train_scaled, MZ_MY_scaled]

## Loading the combined Tower Bending Moment of the downstream turbine 
Input_MZ_MY_Turb2 = scipy.io.loadmat(file_name+'Input_Run_MZ_MY_Turb2.mat')
Input_MZ_MY_Turb2 = Input_MZ_MY_Turb2['Input_Run_Mz_My_Turb2']

yaw_1_train_scaled = Input_MZ_MY_Turb2[0][I1*I2]/30
yaw_2_train_scaled = Input_MZ_MY_Turb2[1][I1*I2]/30
Distance_train_scaled = Input_MZ_MY_Turb2[2][I1*I2]/np.max(Input_MZ_MY_Turb2[2])
MZ_MY_scaled = Input_MZ_MY_Turb2[3][I1*I2]/np.max(Input_MZ_MY_Turb2[3])

Input_MZ_MY_Turb2 = None
Input_MZ_MY_Turb2 = [yaw_1_train_scaled, yaw_2_train_scaled, Distance_train_scaled, MZ_MY_scaled]

Dist_max = np.max(Input_Power2[2])

########## Creating the individual surrogate models for the power and the DEL of the upstream and downstream turbine ###############################
####################################################################################################################################################

# Creating the surrogate model for the power of the upstream turbine
distribution1 = cp.J(cp.Normal(0, 4.95/30))
orthogonal_expansion_ttr1 = cp.orth_ttr(3, distribution1 ) 

Matrix = []
Input = Input_Power1;
for drand in range(0, 9):
    I_rand = random.sample(range(1, len(Input[0])), int(0.9*len(Input[0])))
    approx_model_ttr_Power1  = cp.fit_regression(orthogonal_expansion_ttr1, [Input[0][I_rand]],Input[1][I_rand])
    Coefs = []
    for dq in range(0, len(approx_model_ttr_Power1.keys)):
        a = approx_model_ttr_Power1.A[approx_model_ttr_Power1.keys[dq]]
        Coefs.append(a.tolist())
    Matrix.append(Coefs)
Coefs = np.mean(Matrix, axis = 0)
for dq in range(0, len(approx_model_ttr_Power1.keys)):
    approx_model_ttr_Power1.A[approx_model_ttr_Power1.keys[dq]] = Coefs[dq]

# Creating the surrogate model for the power of the downstream turbine
distribution2 = cp.J(cp.Normal(0, 4.95/30), cp.Normal(0, 4.95/30), cp.Uniform(0, 1))
orthogonal_expansion_ttr2 = cp.orth_ttr(4, distribution2 )

Input = Input_Power2;
Matrix = []
for drand in range(0, 9):
    I_rand = random.sample(range(1, len(Input[0])), int(0.9*len(Input[0])))
    approx_model_ttr_Power2  = cp.fit_regression(orthogonal_expansion_ttr2, [Input[0][I_rand],Input[1][I_rand],Input[2][I_rand]], Input[3][I_rand])
    Coefs = []
    for dq in range(0, len(approx_model_ttr_Power2.keys)):
        a = approx_model_ttr_Power2.A[approx_model_ttr_Power2.keys[dq]]
        Coefs.append(a.tolist())
    Matrix.append(Coefs)
Coefs = np.mean(Matrix, axis = 0)
for dq in range(0, len(approx_model_ttr_Power2.keys)):
    approx_model_ttr_Power2.A[approx_model_ttr_Power2.keys[dq]] = Coefs[dq]


# Creating the surrogate model for the flapwise bending moment of the upstream turbine
orthogonal_expansion_ttr1 = cp.orth_ttr(3, distribution1 ) 

Matrix = []
Input = Input_FlapM_Turb1;
for drand in range(0, 9):
    I_rand = random.sample(range(1, len(Input[0])), int(0.9*len(Input[0])))
    approx_model_ttr_FlapM_Turb1  = cp.fit_regression(orthogonal_expansion_ttr1, [Input[0][I_rand]],Input[1][I_rand])
    Coefs = []
    for dq in range(0, len(approx_model_ttr_FlapM_Turb1.keys)):
        a = approx_model_ttr_FlapM_Turb1.A[approx_model_ttr_FlapM_Turb1.keys[dq]]
        Coefs.append(a.tolist())
    Matrix.append(Coefs)
Coefs = np.mean(Matrix, axis = 0)
for dq in range(0, len(approx_model_ttr_FlapM_Turb1.keys)):
    approx_model_ttr_FlapM_Turb1.A[approx_model_ttr_FlapM_Turb1.keys[dq]] = Coefs[dq]

# Creating the surrogate model for the flapwise bending moment of the downstream turbine
orthogonal_expansion_ttr2 = cp.orth_ttr(3, distribution2 )

Matrix = []
Input = Input_FlapM_Turb2;
for drand in range(0, 9):
    I_rand = random.sample(range(1, len(Input[0])), int(0.9*len(Input[0])))
    approx_model_ttr_FlapM_Turb2  = cp.fit_regression(orthogonal_expansion_ttr2, [Input[0][I_rand],Input[1][I_rand],Input[2][I_rand]], Input[3][I_rand])
    Coefs = []
    for dq in range(0, len(approx_model_ttr_FlapM_Turb2.keys)):
        a = approx_model_ttr_FlapM_Turb2.A[approx_model_ttr_FlapM_Turb2.keys[dq]]
        Coefs.append(a.tolist())
    Matrix.append(Coefs)
Coefs = np.mean(Matrix, axis = 0)
for dq in range(0, len(approx_model_ttr_FlapM_Turb2.keys)):
    approx_model_ttr_FlapM_Turb2.A[approx_model_ttr_FlapM_Turb2.keys[dq]] = Coefs[dq]

# Creating the surrogate model for the combined tower bending moment of the upstream turbine
orthogonal_expansion_ttr1 = cp.orth_ttr(3, distribution1 )

Matrix = []
Input = Input_MZ_MY_Turb1;
for drand in range(0, 9):
    I_rand = random.sample(range(1, len(Input[0])), int(0.9*len(Input[0])))
    approx_model_ttr_MZ_MY_Turb1  = cp.fit_regression(orthogonal_expansion_ttr1, [Input[0][I_rand]],Input[1][I_rand])
    Coefs = []
    for dq in range(0, len(approx_model_ttr_MZ_MY_Turb1.keys)):
        a = approx_model_ttr_MZ_MY_Turb1.A[approx_model_ttr_MZ_MY_Turb1.keys[dq]]
        Coefs.append(a.tolist())
    Matrix.append(Coefs)
Coefs = np.mean(Matrix, axis = 0)
for dq in range(0, len(approx_model_ttr_MZ_MY_Turb1.keys)):
    approx_model_ttr_MZ_MY_Turb1.A[approx_model_ttr_MZ_MY_Turb1.keys[dq]] = Coefs[dq]


# Creating the surrogate model for the combined tower bending moment of the downstream turbine
orthogonal_expansion_ttr2 = cp.orth_ttr(3, distribution2 ) 

Matrix = []
Input = Input_MZ_MY_Turb2;
for drand in range(0, 9):
    I_rand = random.sample(range(1, len(Input[0])), int(0.9*len(Input[0])))
    approx_model_ttr_MZ_MY_Turb2  = cp.fit_regression(orthogonal_expansion_ttr2, [Input[0][I_rand],Input[1][I_rand],Input[2][I_rand]], Input[3][I_rand])
    Coefs = []
    for dq in range(0, len(approx_model_ttr_MZ_MY_Turb2.keys)):
        a = approx_model_ttr_MZ_MY_Turb2.A[approx_model_ttr_MZ_MY_Turb2.keys[dq]]
        Coefs.append(a.tolist())
    Matrix.append(Coefs)
Coefs = np.mean(Matrix, axis = 0)
for dq in range(0, len(approx_model_ttr_MZ_MY_Turb2.keys)):
    approx_model_ttr_MZ_MY_Turb2.A[approx_model_ttr_MZ_MY_Turb2.keys[dq]] = Coefs[dq]

########## Performing the optimization with different assigned weights #############################################################################
####################################################################################################################################################

weights =[]
for dw in range(0,11):
    weights.append([dw/10,(1-dw/10)/4,(1-dw/10)/4, (1-dw/10)/4, (1-dw/10)/4])

Cone_Angle = "_C0"

dist_vec = np.linspace(4, 18, 15)
yw1_list_ttr =[]
yw2_list_ttr =[]
yw1_angle_list_ttr = []
yw1 = np.linspace(-1.167, 1, 500)
yw2 = np.linspace(-1, 1, 500)
yw1_angle = np.linspace(-35, 30, 500)
for dyw1 in range(0, len(yw1)-1):
    for dyw2 in range(0, len(yw2)-1):
        yw1_list_ttr.append(yw1[dyw1])
        yw2_list_ttr.append(yw2[dyw2])
        yw1_angle_list_ttr.append(yw1_angle[dyw1])

for dw in range(0, len(weights)):
    print(dw)
    Value = "W" + str(dw)
    cost_model = (approx_model_ttr_Power1 + approx_model_ttr_Power2)*weights[dw][0] + \
                 (1-approx_model_ttr_FlapM_Turb1)*weights[dw][1] + \
                 (1-approx_model_ttr_FlapM_Turb2)*weights[dw][2] + \
                 (1-approx_model_ttr_MZ_MY_Turb1)*weights[dw][3] + \
                 (1-approx_model_ttr_MZ_MY_Turb2)*weights[dw][4];
    
    Yaw1_Opt_ttr_p5_all_name = "Cost_Model_Yaw1_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle
    Yaw2_Opt_ttr_p5_all_name = "Cost_Model_Yaw2_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle

    Power_Opt_ttr_p5_all_name = "Cost_Model_Power_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle
    FlapM_Turb1_Opt_ttr_p5_all_name = "Cost_Model_FlapM_Turb1_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle
    FlapM_Turb2_Opt_ttr_p5_all_name = "Cost_Model_FlapM_Turb2_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle
    MZ_MY_Turb1_Opt_ttr_p5_all_name = "Cost_Model_MZ_MY_Turb1_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle
    MZ_MY_Turb2_Opt_ttr_p5_all_name = "Cost_Model_MZ_MY_Turb2_Opt_ttr_" + Value + "_p5_all"  + Cone_Angle
    

    Yaw1_Opt_ttr = []
    Yaw2_Opt_ttr = []
    Dist_Opt_ttr = []
    Power_Opt_ttr =[]
    FlapM_Turb1_Opt_ttr =[]
    FlapM_Turb2_Opt_ttr =[]
    MZ_MY_Turb1_Opt_ttr =[]
    MZ_MY_Turb2_Opt_ttr =[]
    
    for ddist in range(0, 15):
        dist = dist_vec[ddist]

        val_ttr = cost_model(yw1_list_ttr, yw2_list_ttr, dist/Dist_max)   
        val_ttr= val_ttr.tolist()
        
        idx_ttr = val_ttr.index(max(val_ttr))
        Yaw1_Opt_ttr = np.append(Yaw1_Opt_ttr, [yw1_list_ttr[idx_ttr]*30])
        Yaw2_Opt_ttr = np.append(Yaw2_Opt_ttr, [yw2_list_ttr[idx_ttr]*30])
        Dist_Opt_ttr = np.append(Dist_Opt_ttr, dist)

        Power_Opt_ttr = np.append(Power_Opt_ttr, (approx_model_ttr_Power2([yw1_list_ttr[idx_ttr]], [yw2_list_ttr[idx_ttr]], dist/Dist_max) + approx_model_ttr_Power1([yw1_list_ttr[idx_ttr]]))/(approx_model_ttr_Power1(0)+approx_model_ttr_Power2(0, 0, dist/Dist_max))) 
        
        FlapM_Turb1_Opt_ttr = np.append(FlapM_Turb1_Opt_ttr, approx_model_ttr_FlapM_Turb1([yw1_list_ttr[idx_ttr]], [yw2_list_ttr[idx_ttr]], dist/Dist_max)/approx_model_ttr_FlapM_Turb1(0, 0, dist/Dist_max))
        FlapM_Turb2_Opt_ttr = np.append(FlapM_Turb2_Opt_ttr, approx_model_ttr_FlapM_Turb2([yw1_list_ttr[idx_ttr]], [yw2_list_ttr[idx_ttr]], dist/Dist_max)/approx_model_ttr_FlapM_Turb2(0, 0, dist/Dist_max))

        MZ_MY_Turb1_Opt_ttr = np.append(MZ_MY_Turb1_Opt_ttr, approx_model_ttr_MZ_MY_Turb1([yw1_list_ttr[idx_ttr]], [yw2_list_ttr[idx_ttr]], dist/Dist_max)/approx_model_ttr_MZ_MY_Turb1(0, 0, dist/Dist_max))
        MZ_MY_Turb2_Opt_ttr = np.append(MZ_MY_Turb2_Opt_ttr, approx_model_ttr_MZ_MY_Turb2([yw1_list_ttr[idx_ttr]], [yw2_list_ttr[idx_ttr]], dist/Dist_max)/approx_model_ttr_MZ_MY_Turb2(0, 0, dist/Dist_max))
        
    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + Yaw1_Opt_ttr_p5_all_name, mdict={Yaw1_Opt_ttr_p5_all_name: Yaw1_Opt_ttr})        
    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + Yaw2_Opt_ttr_p5_all_name, mdict={Yaw2_Opt_ttr_p5_all_name: Yaw2_Opt_ttr})

    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + Power_Opt_ttr_p5_all_name, mdict={Power_Opt_ttr_p5_all_name: Power_Opt_ttr})
    
    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + FlapM_Turb1_Opt_ttr_p5_all_name, mdict={FlapM_Turb1_Opt_ttr_p5_all_name: FlapM_Turb1_Opt_ttr})
    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + FlapM_Turb2_Opt_ttr_p5_all_name, mdict={FlapM_Turb2_Opt_ttr_p5_all_name: FlapM_Turb2_Opt_ttr})

    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + MZ_MY_Turb1_Opt_ttr_p5_all_name, mdict={MZ_MY_Turb1_Opt_ttr_p5_all_name: MZ_MY_Turb1_Opt_ttr})
    scipy.io.savemat('\\\smb.uni-oldenburg.de\\hpc_data\\Back_Up_Asus_Vivobook_15_N580V\\Thesis\\SWiFT\\' + MZ_MY_Turb2_Opt_ttr_p5_all_name, mdict={MZ_MY_Turb2_Opt_ttr_p5_all_name: MZ_MY_Turb2_Opt_ttr})


