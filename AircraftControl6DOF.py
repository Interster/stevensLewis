#%%
# Nota:
# By invalshoeke alpha bo 250 grade hou die interpolasie op met werk.
# Dalk is dit nie 'n probleem as gelykvlug reg is nie.  Maak net seker.
# Intussen, met gelykvlug, maak 'n begrensingsalgoritme vir die alpha.

#%%
import numpy as np
import math as math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%%
# Atmosferiese model

def adc(VT, ALT, AMACH):
    R0 = 2.377e-3 # Sea level density
    TFAC = 1.0 - 0.703e-5*ALT
    
    T = 519*TFAC

    if ALT > 35000.0:
        T = 390.0 # Temperature
    
    RHO = R0*(TFAC**4.14) # Density
    AMACH = VT/(1.4*1716.3*T)**0.5 # Mach number
    QBAR = 0.5*RHO*VT**2 # Dynamic pressure

    PS = 1715.0*RHO*T # Static pressure

    return QBAR

#%%
# Computer model of an F-16

# Definieer alle konstantes met 'n dictionary
vliegtuig = {'S' : 300.0,
             'B' : 30,
             'CBAR' : 11.32,
             'RM' : 1.57e-3,
             'XCGR' : 0.35,
             'HE' : 160.0,
             'Jx' : 9496,
             'Jy' : 55814,
             'Jz' : 63100,
             'Jxy' : 0,
             'Jxz' : 982,
             'Jyz' : 0,
             'G' : 32.17,
             'RTOD' : 57.29578
             }

# Calculate the inertia constants
# Equation 2.4-6, page 80
Jx = vliegtuig['Jx']
Jy = vliegtuig['Jy']
Jz = vliegtuig['Jy']
Jxy = vliegtuig['Jxy']
Jxz = vliegtuig['Jxz']

GAMMA = Jx*Jz - Jxz**2
vliegtuig['C1'] = ((Jy - Jz)*Jz - Jxz**2)/GAMMA
vliegtuig['C2'] = (Jx - Jy + Jz)*Jxz/GAMMA
vliegtuig['C3'] = Jz/GAMMA
vliegtuig['C4'] = Jxz/GAMMA
vliegtuig['C5'] = (Jz - Jx)/Jy
vliegtuig['C6'] = Jxz/Jy
vliegtuig['C7'] = 1/Jy
vliegtuig['C8'] = (Jx*(Jx - Jy) + Jxz**2)/GAMMA
vliegtuig['C9'] = Jx/GAMMA

print(f"{'c1 : '}{vliegtuig['C1']}")
print(f"{'c2 : '}{vliegtuig['C2']}")
print(f"{'c3 : '}{vliegtuig['C3']}")
print(f"{'c4 : '}{vliegtuig['C4']}")
print(f"{'c5 : '}{vliegtuig['C5']}")
print(f"{'c6 : '}{vliegtuig['C6']}")
print(f"{'c7 : '}{vliegtuig['C7']}")
print(f"{'c8 : '}{vliegtuig['C8']}")
print(f"{'c9 : '}{vliegtuig['C9']}")


def TGEAR(THTL):
    if THTL <= 0.77:
        TGEAR = 64.94*THTL
    else:
        TGEAR = 217.38*THTL - 117.38
    
    return TGEAR

def PDOT(P3, P1):
    # PDOT = rate of change of power
    if P1 >= 50.0: 
        # P3 = actual power, P1 = power command
        if P3 >= 50.0:
            T = 5.0
            P2 = P1
        else:
            P2 = 60.0
            T = RTAU(P2 - P3)
    else:
        if P3 >= 50.0:
            T = 5.0
            P2 = 40.0
        else:
            P2 = P1
            T = RTAU(P2 - P3)

    PDOT = T*(P2 - P3)

    return PDOT

def RTAU(DP):
    # Used by function PDOT
    if DP <= 25.0:
        # Reciprocal time constant
        RTAU = 1.0
    elif DP >= 50.0:
        RTAU = 0.1
    else:
        RTAU = 1.9 - 0.36*DP
    
    return RTAU



print(RTAU(26))
print(PDOT(50, 20))
print(TGEAR(0.95))
        





def THRUST(POW, ALT, RMACH):
    # Engine thrust model
    A = [[1060.0, 670.0, 880.0, 1140.0, 1500.0, 1860.0],
         [635.0, 425.0, 690.0, 1010.0, 1330.0, 1700.0],
         [60.0, 25.0, 435.0, 755.0, 1130.0, 1525.0],
         [-1020.0, -710.0, -300.0, 350.0, 910.0, 1360.0],
         [-2700.0, -1900.0, -1300.0, -247.0, 600.0, 1100.0],
         [-3600.0, -1400.0, -595.0, -342.0, -200.0, 700.0]]
    # mil data now
    B = [[12680.0, 9150.0, 6200.0, 3950.0, 2450.0, 1400.0],
         [12680.0, 9150.0, 6313.0, 4040.0, 2470.0, 1400.0],
         [12610.0, 9312.0, 6610.0, 4290.0, 2600.0, 1560.0],
         [12640.0, 9839.0, 7090.0, 4660.0, 2840.0, 1660.0],
         [12390.0, 10176.0, 7750.0, 5320.0, 3250.0, 1930.0],
         [11680.0, 9848.0, 8050.0, 6100.0, 3800.0, 2310.0]]
    # max data now
    C = [[20000.0, 15000.0, 10800.0, 7000.0, 4000.0, 2500.0],
         [21420.0, 15700.0, 11225.0, 7323.0, 4435.0, 2600.0],
         [22700.0, 16860.0, 12250.0, 8154.0, 5000.0, 2835.0],
         [24240.0, 18910.0, 13760.0, 9285.0, 5700.0, 3215.0],
         [26070.0, 21075.0, 15975.0, 11115.0, 6860.0, 3950.0],
         [28886.0, 23319.0, 18300.0, 13484.0, 8642.0, 5057.0]]
    
    H = 0.0001*ALT
    I = int(H)
    if I >= 5:
        I = 4
    DH = H - I
    RM = 5.0*RMACH
    M = int(RM)
    if M >= 5:
        M = 4
    DM = RM - M
    CDH = 1.0 - DH

    S = B[M][I]*CDH + B[M][I + 1]*DH
    T = B[M + 1][I]*CDH + B[M + 1][I + 1]*DH
    TMIL = S + (T - S)*DM

    if POW < 50.0:
        S = A[M][I]*CDH + A[M][I + 1]*DH
        T = A[M + 1][I]*CDH + A[M + 1][I + 1]*DH
        TIDL = S + (T - S)*DM
        THRUST = TIDL + (TMIL - TIDL)*POW*0.02
    else:
        S = C[M][I]*CDH + C[M][I + 1]*DH
        T = C[M + 1][I]*CDH + C[M + 1][I + 1]*DH
        TMAX = S + (T - S)*DM
        THRUST = TMIL + (TMAX - TMIL)*(POW - 50.0)*0.02
    
    return THRUST

print(THRUST(100, 0, 0.8))


def DAMP(ALPHA):
    # Various damping coefficients
    # D[0] -> CXq
    # D[1] -> CYr
    # D[2] -> CYp
    # D[3] -> CZq
    # D[4] -> Clr
    # D[5] -> Clp
    # D[6] -> Cmq
    # D[7] -> Cnr
    # D[8] -> Cnp

    A = [[-0.267, -0.110, 0.308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21],
         [0.882, 0.852, 0.876, 0.958, 0.962, 0.974, 0.819, 0.483, 0.590, 1.21, -0.493, -1.04],
         [-0.108, -0.108, -0.188, 0.110, 0.258, 0.226, 0.344, 0.362, 0.611, 0.529, 0.298, -2.27],
         [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3],
         [-0.126, -0.026, 0.063, 0.113, 0.208, 0.230, 0.319, 0.437, 0.680, 0.100, 0.447, -0.330],
         [-0.360, -0.359, -0.443, -0.420, -0.383, -0.375, -0.329, -0.294, -0.230, -0.210, -0.120, -0.100],
         [-7.21, -0.540, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00],
         [-0.38, -0.363, -0.378, -0.386, -0.370, -0.453, -0.550, -0.582, -0.595, -0.637, -1.02, -0.840],
         [0.061, 0.052, 0.052, -0.012, -0.013, -0.024, -0.050, 0.150, 0.130, 0.158, 0.240, 0.150]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))

    D = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for teller in range(0,8):
        D[teller] = A[teller][2 + K]

    return D


print(DAMP(-10))
damplist = []
alphalist = []
for teller in range(-20, 50):
    alphalist.append(teller)
    damptemp = DAMP(teller)
    damplist.append(damptemp[3])

plt.plot(alphalist, damplist, 'b', label=r'$Damp$')




def CX(ALPHA, EL):
    # x-axis aerodynamic force coefficients
    A = [[-0.099, -0.081, -0.081, -0.063, -0.025, 0.044, 0.097, 0.113, 0.145, 0.167, 0.174, 0.166],
         [-0.048, -0.038, -0.040, -0.021, 0.016, 0.083, 0.127, 0.137, 0.162, 0.177, 0.179, 0.167],
         [-0.022, -0.020, -0.021, -0.004, 0.032, 0.094, 0.128, 0.130, 0.154, 0.161, 0.155, 0.138],
         [-0.040, -0.038, -0.039, -0.025, 0.006, 0.062, 0.087, 0.085, 0.100, 0.110, 0.104, 0.091],
         [-0.083, -0.073, -0.076, -0.072, -0.046, 0.012, 0.024, 0.025, 0.043, 0.053, 0.047, 0.04]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = EL/12.0
    M = int(S)
    if M <= -2:
        M = -1
    if M >= 2:
        M = 1
    DE = S - M
    N = M + int(math.copysign(1.1, DE))
    T = A[M + 2][K + 2]
    U = A[N + 2][K + 2]
    V = T + abs(DA)*(A[M + 2][L + 2] - T)
    W = U + abs(DA)*(A[N + 2][L + 2] - U)
    CX = V + (W - V)*abs(DE)

    return CX

print(CX(0, 0))
cxlist = []
alphalist = []
for teller in range(-50, 80):
    alphalist.append(teller)
    cxlist.append(CX(teller, 0))

plt.plot(alphalist, cxlist, 'b', label=r'$C_x$')

cxlist = []
alphalist = []
for teller in range(-50, 80):
    alphalist.append(teller)
    cxlist.append(CX(teller, 30))

plt.plot(alphalist, cxlist, 'r', label=r'$C_x 25^\circ$')

 
def CY(BETA, AIL, RDR):
    # Sideforce coefficient
    CY = -0.02*BETA + 0.021*(AIL/20.0) + 0.086*(RDR/30.0)

    return CY


def CZ(ALPHA, BETA, EL):
    A = [0.770, 0.241, -0.100, -0.416, -0.731, -1.053, -1.366, -1.646, -1.917, -2.120, -2.248, -2.229]

    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = A[K + 2] + abs(DA)*(A[L + 2] - A[K + 2])

    CZ = S*(1 - (BETA/57.3)**2) - 0.19*(EL/25.0)

    return CZ

print(CZ(0, 0, 0))
czlist = []
alphalist = []
for teller in range(-50, 80):
    alphalist.append(teller)
    czlist.append(CZ(teller, 0, 0))

plt.plot(alphalist, czlist, 'b', label=r'$C_z$')


def CM(ALPHA, EL):
    # pitching moment coefficient
    A = [[0.205, 0.168, 0.186, 0.196, 0.213, 0.251, 0.245, 0.238, 0.252, 0.231, 0.198, 0.192],
         [0.081, 0.077, 0.107, 0.110, 0.110, 0.141, 0.127, 0.119, 0.133, 0.108, 0.081, 0.093],
         [-0.046, -0.020, -0.009, -0.005, -0.006, -0.010, -0.006, -0.001, 0.014, 0.000, -0.013, 0.032],
         [-0.174, -0.145, -0.121, -0.127, -0.129, -0.102, -0.097, -0.113, -0.087, -0.084, -0.069, -0.006],
         [-0.259, -0.202, -0.184, -0.193, -0.199, -0.150, -0.160, -0.167, -0.104, -0.076, -0.041, -0.005]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = EL/12.0
    M = int(S)
    if M <= -2:
        M = -1
    if M >= 2:
        M = 1
    DE = S - M
    N = M + int(math.copysign(1.1, DE))
    T = A[M + 2][K + 2]
    U = A[N + 2][K + 2]
    V = T + abs(DA)*(A[M + 2][L + 2] - T)
    W = U + abs(DA)*(A[N + 2][L + 2] - U)
    CM = V + (W - V)*abs(DE)

    return CM

print(CM(0, 0))

cmlist = []
alphalist = []
for teller in range(-20, 50):
    alphalist.append(teller)
    cmlist.append(CM(teller, 0))

plt.plot(alphalist, cmlist, 'b', label=r'$C_M$')

cmlist = []
alphalist = []
for teller in range(-20, 50):
    alphalist.append(teller)
    cmlist.append(CM(teller, 25))

plt.plot(alphalist, cmlist, 'r', label=r'$C_M 25^\circ$')



def CL(ALPHA, BETA):
    # rolling moment coefficient
    A = [[-0.001, -0.004, -0.008, -0.012, -0.016, -0.019, -0.020, -0.020, -0.015, -0.008, -0.013, -0.015],
         [-0.003, -0.009, -0.017, -0.024, -0.030, -0.034, -0.040, -0.037, -0.016, -0.002, -0.010, -0.019],
         [-0.001, -0.010, -0.020, -0.030, -0.039, -0.044, -0.050, -0.049, -0.023, -0.006, -0.014, -0.027],
         [0.000, -0.010, -0.022, -0.034, -0.047, -0.046, -0.059, -0.061, -0.033, -0.036, -0.035, -0.035],
         [0.007, -0.010, -0.023, -0.034, -0.049, -0.046, -0.068, -0.071, -0.060, -0.058, -0.062, -0.059],
         [0.009, -0.011, -0.023, -0.037, -0.050, -0.047, -0.074, -0.079, -0.091, -0.076, -0.077, -0.076]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = 0.2*abs(BETA)
    M = int(S)
    if M == 0:
        M = 1
    if M >= 6:
        M = 5
    DB = S - M
    N = M + int(math.copysign(1.1, DB))
    T = A[M][K + 2]
    U = A[N][K + 2]
    V = T + abs(DA)*(A[M][L + 2] - T)
    W = U + abs(DA)*(A[N][L + 2] - U)
    DUM = V + (W - V)*abs(DB)
    CL = DUM*math.copysign(1.0, BETA)

    return CL

print(CL(0, 0))

cllist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    cllist.append(CL(teller, 0))

plt.plot(alphalist, cllist, 'b', label=r'$C_L$')






def CN(ALPHA, BETA):
    # yawing moment coefficient
    A = [[0.018, 0.019, 0.018, 0.019, 0.019, 0.018, 0.013, 0.007, 0.004, -0.014, -0.017, -0.033],
         [0.038, 0.042, 0.042, 0.042, 0.043, 0.039, 0.030, 0.017, 0.004, -0.035, -0.047, -0.057],
         [0.056, 0.057, 0.059, 0.058, 0.058, 0.053, 0.032, 0.012, 0.002, -0.046, -0.071, -0.073],
         [0.064, 0.077, 0.076, 0.074, 0.073, 0.057, 0.029, 0.007, 0.012, -0.034, -0.065, -0.041],
         [0.074, 0.086, 0.093, 0.089, 0.080, 0.062, 0.049, 0.022, 0.028, -0.012, -0.002, -0.013],
         [0.079, 0.090, 0.106, 0.106, 0.096, 0.080, 0.068, 0.030, 0.064, 0.015, 0.011, -0.11]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = 0.2*abs(BETA)
    M = int(S)
    if M <= 0:
        M = 1
    if M >= 6:
        M = 5
    DB = S - M
    N = M + int(math.copysign(1.1, DB))
    T = A[M][K + 2]
    U = A[N][K + 2]
    V = T + abs(DA)*(A[M][L + 2] - T)
    W = U + abs(DA)*(A[N][L + 2] - U)
    DUM = V + (W - V)*abs(DB)
    CN = DUM*math.copysign(1.0, BETA)

    return CN

print(CN(-10, -20))

cnlist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    cnlist.append(CN(teller, 0))

plt.plot(alphalist, cnlist, 'b', label=r'$C_N$')



def DLDA(ALPHA, BETA):
    # rolling moment due to ailerons
    A = [[-0.041, -0.052, -0.053, -0.056, -0.050, -0.056, -0.082, -0.059, -0.042, -0.038, -0.027, -0.017],
         [-0.041, -0.053, -0.053, -0.053, -0.050, -0.051, -0.066, -0.043, -0.038, -0.027, -0.023, -0.016],
         [-0.042, -0.053, -0.052, -0.051, -0.049, -0.049, -0.043, -0.035, -0.026, -0.016, -0.018, -0.014],
         [-0.040, -0.052, -0.051, -0.052, -0.048, -0.048, -0.042, -0.037, -0.031, -0.026, -0.017, -0.012],
         [-0.043, -0.049, -0.048, -0.049, -0.043, -0.042, -0.042, -0.036, -0.025, -0.021, -0.016, -0.011],
         [-0.044, -0.048, -0.048, -0.047, -0.042, -0.041, -0.020, -0.028, -0.013, -0.014, -0.011, -0.010],
         [-0.043, -0.049, -0.047, -0.045, -0.042, -0.037, -0.003, -0.013, -0.010, -0.003, -0.007, -0.008]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = 0.1*BETA
    M = int(S)
    if M <= -3:
        M = -2
    if M >= 3:
        M = 2
    DB = S - M
    N = M + int(math.copysign(1.1, DB))
    T = A[M + 3][K + 2]
    U = A[N + 3][K + 2]
    V = T + abs(DA)*(A[M + 3][L + 2] - T)
    W = U + abs(DA)*(A[N + 3][L + 2] - U)
    DLDA = V + (W - V)*abs(DB)

    return DLDA

print(DLDA(-10, 0))

dldalist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dldalist.append(DLDA(teller, 0))

plt.plot(alphalist, dldalist, 'b', label=r'$dLdail$')




def DLDR(ALPHA, BETA):
    # rolling moment due to rudder
    A = [[0.005, 0.017, 0.014, 0.010, -0.005, 0.009, 0.019, 0.005, -0.000, -0.005, -0.011, 0.008],
         [0.007, 0.016, 0.014, 0.014, 0.013, 0.009, 0.012, 0.005, 0.000, 0.004, 0.009, 0.007],
         [0.013, 0.013, 0.011, 0.012, 0.011, 0.009, 0.008, 0.005, -0.002, 0.005, 0.003, 0.005],
         [0.018, 0.015, 0.015, 0.014, 0.014, 0.014, 0.014, 0.015, 0.013, 0.011, 0.006, 0.001],
         [0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.011, 0.010, 0.008, 0.008, 0.007, 0.003],
         [0.021, 0.011, 0.010, 0.011, 0.010, 0.009, 0.008, 0.010, 0.006, 0.005, 0.000, 0.001],
         [0.023, 0.010, 0.011, 0.011, 0.011, 0.010, 0.008, 0.010, 0.006, 0.014, 0.020, 0.000]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = 0.1*BETA
    M = int(S)
    if M <= -3:
        M = -2
    if M >= 3:
        M = 2
    DB = S - M
    N = M + int(math.copysign(1.1, DB))
    T = A[M + 3][K + 2]
    U = A[N + 3][K + 2]
    V = T + abs(DA)*(A[M + 3][L + 2] - T)
    W = U + abs(DA)*(A[N + 3][L + 2] - U)
    DLDR = V + (W - V)*abs(DB)

    return DLDR

print(DLDR(-10, 0))

dldrlist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dldrlist.append(DLDR(teller, 0))

plt.plot(alphalist, dldrlist, 'b', label=r'$dLdr$')


def DNDA(ALPHA, BETA):
    # yawing moment due to ailerons
    A = [[0.001, -0.027, -0.017, -0.013, -0.012, -0.016, 0.001, 0.017, 0.011, 0.017, 0.008, 0.016],
         [0.002, -0.014, -0.016, -0.016, -0.014, -0.019, -0.021, 0.002, 0.012, 0.016, 0.015, 0.011],
         [-0.006, -0.008, -0.006, -0.006, -0.005, -0.008, -0.005, 0.007, 0.004, 0.007, 0.006, 0.006],
         [-0.011, -0.011, -0.010, -0.009, -0.008, -0.006, -0.000, 0.004, 0.007, 0.010, 0.004, 0.010],
         [-0.015, -0.015, -0.014, -0.012, -0.011, -0.008, -0.002, 0.002, 0.006, 0.012, 0.011, 0.011],
         [-0.024, -0.010, -0.004, -0.002, -0.001, 0.003, 0.014, 0.006, -0.001, 0.004, 0.004, 0.006],
         [-0.022, 0.002, -0.003, -0.005, -0.003, -0.001, -0.009, -0.009, -0.001, 0.003, -0.002, 0.001]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = 0.1*BETA
    M = int(S)
    if M <= -3:
        M = -2
    if M >= 3:
        M = 2
    DB = S - M
    N = M + int(math.copysign(1.1, DB))
    T = A[M + 3][K + 2]
    U = A[N + 3][K + 2]
    V = T + abs(DA)*(A[M + 3][L + 2] - T)
    W = U + abs(DA)*(A[N + 3][L + 2] - U)
    DNDA = V + (W - V)*abs(DB)

    return DNDA

print(DNDA(-10, 0))

dndalist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dndalist.append(DNDA(teller, 0))

plt.plot(alphalist, dndalist, 'b', label=r'$dnda$')


def DNDR(ALPHA, BETA):
    # yawing moment due to rudder
    A = [[-0.018, -0.052 ,-0.052, -0.052, -0.054, -0.049, -0.059, -0.051, -0.030, -0.037, -0.026, -0.013],
         [-0.028, -0.051, -0.043, -0.046, -0.045, -0.049, -0.057, -0.052, -0.030, -0.033, -0.030, -0.008],
         [-0.037, -0.041, -0.038, -0.040, -0.040, -0.038, -0.037, -0.030, -0.027, -0.024, -0.019, -0.013],
         [-0.048, -0.045, -0.045, -0.045, -0.044, -0.045, -0.047, -0.048, -0.049, -0.045, -0.033, -0.016],
         [-0.043, -0.044, -0.041, -0.041, -0.040, -0.038, -0.034, -0.035, -0.035, -0.029, -0.022, -0.009],
         [-0.052, -0.034, -0.036, -0.036, -0.035, -0.028, -0.024, -0.023, -0.020, -0.016, -0.010, -0.014],
         [-0.062, -0.034, -0.027, -0.028, -0.027, -0.027, -0.023, -0.023, -0.019, -0.009, -0.025, -0.010]]
    
    S = 0.2*ALPHA
    K = int(S)
    if K <= -2:
        K = -1
    if K >= 9:
        K = 8
    DA = S - K
    L = K + int(math.copysign(1.1, DA))
    S = 0.1*BETA
    M = int(S)
    if M <= -3:
        M = -2
    if M >= 3:
        M = 2
    DB = S - M
    N = M + int(math.copysign(1.1, DB))
    T = A[M + 3][K + 2]
    U = A[N + 3][K + 2]
    V = T + abs(DA)*(A[M + 3][L + 2] - T)
    W = U + abs(DA)*(A[N + 3][L + 2] - U)
    DNDR = V + (W - V)*abs(DB)

    return DNDR

print(DNDR(-10, 0))

dndrlist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dndrlist.append(DNDR(teller, 0))

plt.plot(alphalist, dndrlist, 'b', label=r'$dndr$')


#%%
# 6DOF weergawe van die vlugsimulasie

def f(x, t, v, THTL, EL, AIL, RDR, AMACH, XCG):
    # Assign state & control variables
    # x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
    VT = x[0] # True airspeed (ft/s)
    ALPHA = x[1]*v['RTOD']
    BETA = x[2]*v['RTOD']
    PHI = x[3]
    THETA = x[4]
    PSI = x[5]
    P = x[6]
    Q = x[7]
    R = x[8]
    ALT = x[11]
    POW = x[12]
    # Air data computer and engine model
    # Bereken dinamiese druk
    QBAR = adc(VT, ALT, AMACH)
    CPOW = TGEAR(THTL)
    xd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Inisialiseer xd
    xd[12] = PDOT(POW, CPOW)
    T = THRUST(POW, ALT, AMACH)
    # Look-up tables and component buildup
    CXT = CX(ALPHA, EL)
    CYT = CY(BETA, AIL, RDR)
    CZT = CZ(ALPHA, BETA, EL)
    DAIL = AIL/20.0
    DRDR = RDR/30.0
    CLT = CL(ALPHA, BETA) + DLDA(ALPHA, BETA)*DAIL + DLDR(ALPHA, BETA)*DRDR
    CMT = CM(ALPHA, EL)
    CNT = CN(ALPHA, BETA) + DNDA(ALPHA, BETA)*DAIL + DNDR(ALPHA, BETA)*DRDR
    # Add damping derivatives:
    TVT = 0.5/VT
    B2V = v['B']*TVT
    CQ = v['CBAR']*Q*TVT
    D = DAMP(ALPHA)
    CXT = CXT + CQ*D[0]
    CYT = CYT + B2V*(D[1]*R + D[2]*P)
    CZT = CZT + CQ*D[3]
    CLT = CLT + B2V*(D[4]*R + D[5]*P)
    CMT = CMT + CQ*D[6] + CZT*(v['XCGR'] - XCG)
    CNT = CNT + B2V*(D[7]*R + D[8]*P) - CYT*(v['XCGR'] - XCG)*v['CBAR']/v['B']
    # Get ready for state equations
    CBTA = math.cos(x[2])
    U = VT*math.cos(x[1])*CBTA
    V = VT*math.sin(x[2])
    W = VT*math.sin(x[1])*CBTA
    STH = math.sin(THETA)
    CTH = math.cos(THETA)
    SPH = math.sin(PHI)
    CPH = math.cos(PHI)
    SPSI = math.sin(PSI)
    CPSI = math.cos(PSI)
    QS = QBAR*v['S']
    QSB = QS*v['B']
    RMQS = v['RM']*QS
    GCTH = v['G']*CTH
    QSPH = Q*SPH
    AY = RMQS*CYT
    AZ = RMQS*CZT
    # Table 2.4-1 Flat earth body axes 6DOF equations, page 81
    # Force equations
    UDOT = R*V - Q*W - v['G']*STH + v['RM']*(QS*CXT + T)
    VDOT = P*W - R*U + GCTH*SPH + AY
    WDOT = Q*U - P*V + GCTH*CPH + AZ
    DUM = (U*U + W*W)
    xd[0] = (U*UDOT + V*VDOT + W*WDOT)/VT
    xd[1] = (U*WDOT - W*UDOT)/DUM
    xd[2] = (VT*VDOT - V*xd[0])*CBTA/DUM

    #Kinematics
    xd[3] = P + (STH/CTH)*(QSPH + R*CPH)
    xd[4] = Q*CPH - R*SPH
    xd[5] = (QSPH + R*CPH)/CTH
    #Moment
    xd[6] = (v['C2']*P + v['C1']*R + v['C4']*v['HE'])*Q + QSB*(v['C3']*CLT + v['C4']*CNT)
    xd[7] = (v['C5']*P - v['C7']*v['HE'])*R + v['C6']*(R*R - P*P) + QS*v['CBAR']*v['C7']*CMT
    xd[8] = (v['C8']*P - v['C2']*R + v['C9']*v['HE'])*Q + QSB*(v['C4']*CLT + v['C9']*CNT)

    # Navigation
    #
    T1 = SPH*CPSI
    T2 = CPH*STH
    T3 = SPH*SPSI
    S1 = CTH*CPSI
    S2 = CTH*SPSI
    S3 = T1*STH - CPH*SPSI
    S4 = T3*STH + CPH*CPSI
    S5 = SPH*CTH
    S6 = T2*CPSI + T3
    S7 = T2*SPSI - T1
    S8 = CPH*CTH
    #
    xd[9] = U*S1 + V*S3 + W*S6 # North speed
    xd[10] = U*S2 + V*S4 + W*S7 # East speed
    xd[11] = U*STH - V*S5 - W*S8 # Vertical speed

    AN = -AZ/v['G']
    ALAT = AY/v['G']

    return xd

# f(x, t, v, THTL, ELEV, AIL, RDR, AMACH, XCG)
# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, ALT, POW]
# Table 3.3-2 F-16 Model test case, page 128
x = [500.0, 0.5, -0.2, -1, 1, -1, 0.7, -0.8, 0.9, 1000, 900, 10000, 90]
xd = f(x, 0, vliegtuig, 0.9, 20.0, -15.0, -20.0, 0.0, 0.4)
print(xd)


#%% Integreer funksie
# Loop die funksie met Runge Kutta integrasie
# f(x, t, v, THTL, ELEV, AIL, RDR, AMACH, XCG)
# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
# # Table 3.3-2 F-16 Model test case, page 128
x0 = [500.0, 0.5, -0.2, -1, 1, -1, 0.7, -0.8, 0.9, 1000, 900, 10000, 90]
THTL = 0.9
EL = 0.0
AIL = 0.0
RDR = 0.0
AMACH = 0.0
XCG = 0.4
print(f(x0, 0, vliegtuig, 0.9, 20.0, -15.0, -20.0, 0.0, 0.4))


t = np.linspace(0, 2.45, 101)
# f(x, t, v, THTL, EL, AIL, RDR, AMACH, XCG):
sol = odeint(f, x0, t, args=(vliegtuig, THTL, EL, AIL, RDR, AMACH, XCG))


plt.plot(t, sol[:, 4]*180/3.14159, 'b', label=r'$\theta (t)$')
plt.plot(t, sol[:, 1]*180/3.14159, 'g', label=r'$\alpha (t)$')

plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.plot(t, sol[:, 0], 'b', label='V(t)')

plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()


#%% 
# Euler integrasie
# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, ALT, POW]
x0 = [500.0, 0.5, -0.2, -1.0, 1.0, -1.0, 0.7, -0.8, 0.9, 1000.0, 900.0, 10000.0, 90.0]
THTL = 0.9
EL = 0.0
AIL = 0.0
RDR = 0.0
AMACH = 0.0
XCG = 0.4
print(f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, AMACH, XCG))

thetaplot = [x0[4]]
t = 0.0

for teller in range(1, 2400):
    xd = f(x0, t, vliegtuig, THTL, EL, AIL, RDR, AMACH, XCG)
    xint = [i * 0.001 for i in xd]
    x0 = [x + y for x, y in zip(x0, xint)]

    thetaplot.append(x0[4])
    t = t + 0.001
    
thetaplotdeg = [i*180/3.14159 for i in thetaplot]
plt.plot(thetaplotdeg, 'b', label=r'$\theta (t)$')

#%%
# Trimmer funksie
# Los op die gelykvlug kondisies vir die vliegtuigmodel


def doelfunksie(inset, vliegtuig, konstant):
    # inset = [throttle, elevator, theta]
    # konstant = [Spoed [ft/s], hoogte [ft]]
    # x = [VT, ALPHA, THETA, Q, H, Distance]
    # As ALPHA = THETA is dit gelykvlug, dus GAMMA = 0
    x0 = [konstant[0], inset[2], inset[2], 0, konstant[1], 0]

    # f(x, t, vliegtuig, land, THTL, ELEV)
    xd = f(x0, 0, vliegtuig, False, inset[0], inset[1])

    doel = xd[0]**2 + 100*xd[1]**2 + 10*xd[3]**2

    return doel


#%%

# Druk opskrifte
print(f"{'Altitude' :^10}{'Speed' :^10}{'Throttle' :^10}{'Elevator' :^10}{'Alpha' :^10}")
print(f"{'ft' :^10}{'ft/s' :^10}{' ' :^10}{'deg' :^10}{'deg' :^10}")


# inset = [throttle, elevator [deg], theta [rad]]
inset = [0.297, -25.7, 22.1*3.14159/180]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [170, 0]
res = minimize(doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")


#%%
# How to interpolate in 2D using numpy and scipy
#
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 13)
y = np.array([0, 2, 3, 3.5, 3.75, 3.875, 3.9375, 4])
X, Y = np.meshgrid(x, y)
Z = np.sin(np.pi*X/2) * np.exp(Y/2)

x2 = np.linspace(0, 4, 65)
y2 = np.linspace(0, 4, 65)
f = interp2d(x, y, Z, kind='cubic')
Z2 = f(x2, y2)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].pcolormesh(X, Y, Z)

X2, Y2 = np.meshgrid(x2, y2)
ax[1].pcolormesh(X2, Y2, Z2)

plt.show()




# %%
