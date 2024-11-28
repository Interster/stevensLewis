#%% Import alle modules
import numpy as np
import math



#%%
# Figure 3.3-1 Computer model of a transport aircraft

# Definieer alle konstantes met 'n dictionary
vliegtuig = {'S' : 2170.0,
             'CBAR' : 17.5,
             'AM' : 5e3,
             'AIYY' : 4.1e6,
             'G' : 32.17,
             'TSTAT' : 6.0e4,
             'DTDV' : -38.0,
             'ZE' : 2.0,
             'CLA' :  0.085,
             'CLADOT' : 0.0,
             'CDCLS' : 0.042,
             'CMA' : -0.022,
             'CMQ' : -16.0,
             'CMADOT' : -6.0,
             'CMDE' : -0.016,
             'RTOD' : 57.29578
             }


# %%

# Atmosferiese model

def adc(VT, H, AMACH, QBAR):
    QBAR = 1

    return QBAR

# Vliegtuig funksie

def f(v, t, x, xd, land, THTL, ELEV):
    VT = x(0) # True airspeed (ft/s)
    ALPHA = x(1)*v['RTOD']
    THETA = x(2)
    Q = x(3)
    H = x(4)

    THETAD = v['RTOD']*THETA
    QD = v['RTOD']*Q

    # Bereken dinamiese druk
    QBAR = adc(VT, H, 0)
    QS = QBAR*S 
    SALP = math.sin(x(1))
    CALP = math.cos(x(1))
    STH = math.sin(THETA)
    CTH = math.cos(THETA)
    GAM = THETA - x(1)

    if land:
        CL0 = 1.0
        CD0 = 0.08
        CM0 = -0.20
        DCDG = 0.02
        DCMG = -0.05
    else:
        CL0 = 0.20
        CD0 = 0.016
        CM0 = 0.05
        DCDG = 0.0
        DCMG = 0.0

    # Stukrag (thrust)
    THR = (v['TSTAT'] + VT*v['DTDV'])*THTL
    # Hefkragkoeffisient (Lift coefficient)
    CL = CL0 + v['CLA']*ALPHA
    # Heimoment (pitch moment)
    CM = DCMG + v['CM0'] + v['CMA']*ALPHA + v['CMDE']*ELEV + CL*(XCG - 0.25)
    # Sleurkrag (drag polar)
    CD = DCDG + CD0 + v['CDCLS']*CL*CL

    # VT rate
    xd[0] = (THR*CALP - QS*CD)/v['AM'] - v['G']*math.sin(GAM)

    TEMP = -THR*SALP - QS*CL + v['AM']*(VT*Q + v['G']*math.cos(GAM))

    # Alpha rate
    xd[1] = TEMP/(v['AM']*VT + QS*v['CLADOT']) 

    # Pitch rate
    xd[2] = Q

    # Damping terms
    D = 0.5*v['CBAR']*(v['CMQ']*Q + v['CMADOT']*x[1])/VT

    # Pitch acceleration
    xd[3] = (QS*v['CBAR']*(CM + D) + THR*v['ZE')/v['AIYY']]

    # Upward speed
    xd[4] = VT*(CALP*STH - SALP*CTH)

    # Horizontal speed
    xd[5] = VT(CALP*CTH + SALP*STH)

    # Normal acceleration
    AN = QS*(v['CL']*CALP + CD*SALP)/(v['G']*v['AM'])

    return xd




#%% Roep funksie

t = 0 # Tyd inisialiseer
# Toestandsveranderlike model (state space model)
x = np.array([1, 2, 3, 4, 5])
# Afgeleide van model
xd = np.array([1, 2, 3, 4, 5])

# Roep die funksie
print(f(vliegtuig, t, x, xd), True, 0, 0)
# %%
