import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg, optimize
from scipy.integrate import solve_ivp


H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
T_min, T_max = 278, 298
H_min, H_max = 42, H_0
Enz_min, Enz_max = 61, 80
s = 5
r = int(s*(s-1)/2)
delta_t = 1


A_H = np.array([-0.10280675, -0.01900384, -1.0640578, -0.2409164, -0.55442332])
A_H = np.array([A_H[0]+delta_t, -A_H[0], A_H[1]+delta_t, -A_H[1], A_H[2]+delta_t, -A_H[2], A_H[3]+delta_t, -A_H[3], A_H[4]+delta_t, -A_H[4]])
B_H = np.array([-0.07503288, -0.01304328, -0.89262608, -0.25439021, -0.2968465])
B_H = np.array([B_H[0], -B_H[0]+delta_t, B_H[1], -B_H[1]+delta_t, B_H[2], -B_H[2]+delta_t, B_H[3], -B_H[3]+delta_t, B_H[4], -B_H[4]+delta_t])
C_H = np.array([-1.1250703,  -0.21467077, -12.75212815,  -3.70708767,  -7.70798855])
C_H = np.array([C_H[0], -C_H[0], C_H[1], -C_H[1], C_H[2], -C_H[2], C_H[3], -C_H[3], C_H[4], -C_H[4]])
D_H = np.array([328.0856564, 61.06533784, 3847.73057284, 1094.81508891, 2286.72706302])
D_H =np.array([D_H[0], -D_H[0], D_H[1], -D_H[1], D_H[2], -D_H[2], D_H[3], -D_H[3], D_H[4], -D_H[4]])
Q_H = np.array([0.07264316, -0.11880306, -0.09235975, -0.21013151,  0.02206956,  0.02047818, 0.05606904,  0.1741325, 0.08299591, -0.10074457])
T_H = np.array([0.08162662, -0.0247667, -0.04874518, -0.1643359,  -0.01855246, -0.12330535, -0.0595362,  0.15179147,  0.02938379, -0.11320038])
R_H = np.array([1.14902717, -0.93113606, -1.03643109, -2.01372202, -0.3875179, -1.31042227, -0.74833271,  1.57561027,  0.91628172, -1.10607559])
S_H = np.array([-336.11696047,  280.33474471,  308.28579863, 609.3326419,   112.83704085, 383.89718463,  217.84698594, -483.75776015, -277.27304113, 337.39388709])


A_Enz = np.array([0.13269727, 0.01619835,  0.02322045,  0.3872233,  -0.09060637])
B_Enz = np.array([0.02053655+1, 0.01204005+1, 0.04211448+1, 0.39663007+1, -0.2807609+1])
C_Enz = np.array([1.90545383, 0.19150447, 0.41481407, 6.89491858, 3.07335709])
D_Enz = np.array([-549.70770684, -54.44110012, -119.76911537, -2045.47431075, -851.52482217])
Q_Enz = np.array([0.09760416,  0.10384821, -0.17141225, -0.21885381,  0.01241944, -0.03658759, -0.13936494, -0.1043193, -0.17073999, 0.05280885])
T_Enz = np.array([0.12639949,  0.05800941, -0.17144736, -0.29612729, -0.10601023, -0.00282248, -0.01091229, -0.00797218, -0.20065254, -0.03887606])
R_Enz = np.array([1.57084236,  1.20726075, -1.3323752,  -1.03936411, -1.02419315, -0.30943017, -0.58911003, -0.88571775, -1.1249749, 0.73588291])
S_Enz = np.array([-459.8515884, -354.62804675, 410.19687503, 335.41741443,  295.20607001, 91.07062292,  176.53935914,  262.51124547,  347.28520893, -215.4568751])

"""
A = A.flatten()
B = B.flatten()
C = C.flatten()
Q = Q.flatten()
T = T.flatten()
R = R.flatten()
"""

print('A_H type:', type(A_H))


def pwa(H, Enz, Ta, A, B, C, D):
    return A*H + B*Enz + C*Ta + D

def get_gmin(s,A,B,C,D):
    gmin = np.empty(s)
    for i in range(s):
        choice = [pwa(H_min,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_min,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_max,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_max,T_max,A[i],B[i],C[i],D[i])]
        gmin[i] = min(choice)
    return gmin


def get_gmax(s,A,B,C,D):
    gmax = np.empty(s)
    for i in range(s):
        choice = [pwa(H_min,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_min,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_min,Enz_max,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_min,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_min,T_max,A[i],B[i],C[i],D[i]),
        pwa(H_max,Enz_max,T_min,A[i],B[i],C[i],D[i]), pwa(H_max,Enz_max,T_max,A[i],B[i],C[i],D[i])]
        gmax[i] = max(choice)
        # print(choice)
    return gmax



print("####################################################")
print("####################################################")
print("###################### check for parameters #######################")
print("####################################################")
print("####################################################")
print("A_H :\n", A_H.shape)


print("####################################################")
print("####################################################")
print("###################### g and h of dH #######################")
print("####################################################")
print("####################################################")
print("gmin:\n", get_gmin(int(A_H.shape[0]),A_H,B_H,C_H,D_H))
print("gmax:\n", get_gmax(int(A_H.shape[0]),A_H,B_H,C_H,D_H))
print("hmin:\n", get_gmin(r,Q_H,T_H,R_H,S_H))
print("hmax:\n", get_gmax(r,Q_H,T_H,R_H,S_H))

print("####################################################")
print("####################################################")
print("###################### g and h of dEnz #######################")
print("####################################################")
print("####################################################")
print("gmin:\n", get_gmin(s,A_Enz,B_Enz,C_Enz,D_Enz))
print("gmax:\n", get_gmax(s,A_Enz,B_Enz,C_Enz,D_Enz))
print("hmin:\n", get_gmin(r,Q_Enz,T_Enz,R_Enz,S_Enz))
print("hmax:\n", get_gmax(r,Q_Enz,T_Enz,R_Enz,S_Enz))