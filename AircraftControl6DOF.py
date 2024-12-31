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
# 6DOF weergawe van die vlugsimulasie

def f(x, t, v, land, THTL, ELEV):
    # Assign state & control variables
    # x = [VT, ALPHA, THETA, Q, H, Distance]
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
    QBAR = adc(VT, H, 0)
    CPOW = TGEAR(THTL)
    xd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Inisialiseer xd
    xd[13] = PDOT(POW, CPOW)
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
    B2V = B*TVT
    CQ = CBAR*Q*TVT
    D = DAMP(ALPHA)
    CXT = CXT + CQ*D[1]
    CYT = CYT + B2V*(D[2]*R + D[3]*P)
    CZT = CZT + CQ*D[4]
    CLT = CLT + B2V*(D[5]*R + D[6]*P)
    CMT = CMT + CQ*D[7] + CZT*(XCGR - XCG)
    CNT = CNT + B2V*(D[8]*R + D[9]*P) - CYT*(XCGR - XCG)*CBAR/B
    # Get ready for state equations
    CBTA = cos(x[2])
    U = VT*cos(x[1])*CBTA
    V = VT*sin(x[2])
    W = VT*sin(x[1])*CBTA
    STH = sin(THETA)
    CTH = cos(THETA)
    SPH = sin(PHI)
    CPH = cos(PHI)
    SPSI = sin(PSI)
    CPSI = cos(PSI)
    QS = QBAR*S
    QSB = QS*B
    RMQS = RM*QS
    GCTH = G*CTH
    QSPH = Q*SPH
    AY = RMQS*CYT
    AZ = RMQS*CZT
    # Table 2.4-1 Flat earth body axes 6DOF equations, page 81
    # Force equations
    UDOT = R*V - Q*W - G*STH + RM*(QS*CXT + T)
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
    xd[6] = (C2*P + C1*R + C4*HE)*Q + QSB*(C3*CLT + C4*CNT)
    xd[7] = (C5*P - C7*HE)*R + C6*(R*R - P*P) + QS*CBAR*C7*CMT
    xd[8] = (C8*P - C2*R + C9*HE)*Q + QSB*(C4*CLT + C9*CNT)

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
    xd[10] = U*S1 + V*S3 + W*S6 # North speed
    xd[11] = U*S2 + V*S4 + W*S7 # East speed
    xd[12] = U*STH - V*S5 - W*S8 # Vertical speed

    AN = -AZ/G
    ALAT = AY/G

    return xd

# f(x, t, v, land, THTL, ELEV)
x = [0,0,0,0,0,0,0,0,0,0,0,0,0]
print(f(x, 0, vliegtuig, True, 0.5, 10.0))


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

#%%
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
        

#%%
# Calculate the inertia constants
# Equation 2.4-6, page 80
Jx = vliegtuig['Jx']
Jy = vliegtuig['Jy']
Jz = vliegtuig['Jy']
Jxy = vliegtuig['Jxy']
Jxz = vliegtuig['Jxz']

GAMMA = Jx*Jz - Jxz**2
c1 = ((Jy - Jz)*Jz - Jxz**2)/GAMMA
c2 = (Jx - Jy + Jz)*Jxz/GAMMA
c3 = Jz/GAMMA
c4 = Jxz/GAMMA
c5 = (Jz - Jx)/Jy
c6 = Jxz/Jy
c7 = 1/Jy
c8 = (Jx*(Jx - Jy) + Jxz**2)/GAMMA
c9 = Jx/GAMMA

print(f"{"c1 : "}{c1}")
print(f"{"c2 : "}{c2}")
print(f"{"c3 : "}{c3}")
print(f"{"c4 : "}{c4}")
print(f"{"c5 : "}{c5}")
print(f"{"c6 : "}{c6}")
print(f"{"c7 : "}{c7}")
print(f"{"c8 : "}{c8}")
print(f"{"c9 : "}{c9}")
# %%
