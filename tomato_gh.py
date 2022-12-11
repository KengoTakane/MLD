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
H_min, H_max = 42, H_0-H_plusinf
Enz_min, Enz_max = 61, 80
s = 5
r = s*(s-1)/2


A = np.array([[-4.21270271e-04+1], [-2.36178784e-04+1], [-3.22164500e-04+1], [-2.52650737e-04+1], [-5.55708636e-04+1]])
B = np.array([[-4.84453301e-05], [-3.14845920e-05], [-5.36849855e-05], [-1.67910305e-05], [1.35079377e-05]])
C = np.array([[6.47483981e-03], [3.51970772e-03], [5.42080197e-03], [4.06639287e-03], [8.64274190e-03]])
D = np.array([-3.29723012e-01, -3.06409958e-01, -3.63390858e-01, -3.42315514e-01, -3.13246241e-01])
Q = np.array([[0.02627317], [-0.00111771], [0.01880834], [-0.01876554], [-0.04472564], [-0.03862955], [-0.05427779], [0.02798923], [-0.00845947], [-0.01385068]])
T = np.array([[-0.07538652], [0.04318842], [-0.04245705], [-0.05633143], [0.05458913], [0.0601666], [-0.03055113], [-0.01195942], [-0.00280735], [0.02021543]])
R = np.array([[-0.16798629], [-0.24951434], [-0.34649457], [0.24132598], [0.2253457], [0.20345188], [0.14436165], [-0.23591518], [0.32142254], [0.1428877]])
S = np.array([8.4387248, 3.37998739, 16.72282594, 24.14807339, 9.62861827, 1.78944618, 51.78167187, -6.80356885, -7.37417206, -0.34880202])

A = A.flatten()
B = B.flatten()
C = C.flatten()
Q = Q.flatten()
T = T.flatten()
R = R.flatten()




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
        gmax[i] = min(choice)
    return gmax


print("gmin:\n", get_gmin(s,A,B,C,D))
print("gmax:\n", get_gmax(s,A,B,C,D))
print("hmin:\n", get_gmin(r,Q,T,R,S))
print("hmax:\n", get_gmax(r,Q,T,R,S))