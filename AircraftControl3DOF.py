#%% Laai modules
import math as math

#%% Figure 3.3-1 Computer model of a transport aircraft

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
             'RTOD' : 57.29578,
             'CL0' : 0.20,
             'CD0' : 0.016,
             'CM0' : 0.05,
             'DCDG' : 0.0,
             'DCMG' : 0.0,
             'XCG' : 0.25
             }


# %% Atmosferiese model

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


#%% Vliegtuig funksie

def f(x, t, v, land, THTL, ELEV):
    # x = [VT, ALPHA, THETA, Q, H, Distance]
    VT = x[0] # True airspeed (ft/s)
    ALPHA = x[1]*v['RTOD']
    THETA = x[2]
    Q = x[3]
    H = x[4]

    THETAD = v['RTOD']*THETA
    QD = v['RTOD']*Q

    # Bereken dinamiese druk
    QBAR = adc(VT, H, 0)
    QS = QBAR*v['S'] 
    SALP = math.sin(x[1])
    CALP = math.cos(x[1])
    STH = math.sin(THETA)
    CTH = math.cos(THETA)
    GAM = THETA - x[1]

    if land:
        v['CL0'] = 1.0
        v['CD0'] = 0.08
        v['CM0'] = -0.20
        v['DCDG'] = 0.02
        v['DCMG'] = -0.05
    else:
        v['CL0'] = 0.20
        v['CD0'] = 0.016
        v['CM0'] = 0.05
        v['DCDG'] = 0.0
        v['DCMG'] = 0.0

    # Stukrag (thrust)
    THR = (v['TSTAT'] + VT*v['DTDV'])*THTL
    # Hefkragkoeffisient (Lift coefficient)
    CL = v['CL0'] + v['CLA']*ALPHA
    # Heimoment (pitch moment)
    CM = v['DCMG'] + v['CM0'] + v['CMA']*ALPHA + v['CMDE']*ELEV + CL*(v['XCG'] - 0.25)
    # Sleurkrag (drag polar)
    CD = v['DCDG'] + v['CD0'] + v['CDCLS']*CL*CL

    xd = [0, 0, 0, 0, 0, 0] # Inisialiseer xd
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
    xd[3] = (QS*v['CBAR']*(CM + D) + THR*v['ZE'])/v['AIYY']

    # Upward speed
    xd[4] = VT*(CALP*STH - SALP*CTH)

    # Horizontal speed
    xd[5] = VT*(CALP*CTH + SALP*STH)

    # Normal acceleration
    AN = QS*(CL*CALP + CD*SALP)/(v['G']*v['AM'])

    return xd


#%% Trimmer funksie
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





# %%