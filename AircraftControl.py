#%% Import alle modules
import numpy as np
import math as math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
             'RTOD' : 57.29578,
             'CL0' : 0.20,
             'CD0' : 0.016,
             'CM0' : 0.05,
             'DCDG' : 0.0,
             'DCMG' : 0.0,
             'XCG' : 0.25
             }


# %%

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

# Vliegtuig funksie

def f(x, t, v, land, THTL, ELEV):
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


#%% 
# f(x, t, v, land, THTL, ELEV)
x = [170,22.1*3.14159/180,22.1*3.14159/180,0,0,0]

print(f(x, 0, vliegtuig, False, 0, 0))

#%% Integreer funksie

#x0 = [spoed, alpha, theta, heiversnelling, 
# opwaartse spoed, horizontale spoed]
x0 = [170,22.1*3.14159/180,22.1*3.14159/180,0,0,0]
throttle = 0.297
elev = -25.7


t = np.linspace(0, 10, 101)
sol = odeint(f, x0, t, args=(vliegtuig, False, throttle, elev))

plt.plot(t, sol[:, 0], 'b', label='V(t)')
plt.plot(t, sol[:, 1], 'g', label='\alpha(t)')

plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#%%
# Trimmer funksie

def doelfunksie(inset, vliegtuig):
    # inset = [Spoed, throttle, elevator, theta]
    x0 = [inset[0], inset[3], inset[3],0,0,0]

    xd = f(x0, 0, vliegtuig, False, inset[1], inset[2])

    doel = xd[0]**2 + 100*xd[1]**2 + 10*xd[3]**2

    return doel

#%%

inset = [170, 0.2, -20, 22.1*3.14159/180]
res = minimize(doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig), options={'xatol': 1e-8, 'disp': True})

print(res.x)

# inset = [Spoed, throttle, elevator, theta]
inset = [500, 0.293, 2.46, 0.58*3.14159/180]
res = minimize(doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig), options={'xatol': 1e-8, 'disp': True})

print(res.x)

#%%
# Voorbeeld integrasie
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]

    return dydt

b = 0.25
c = 5.0

y0 = [np.pi - 0.1, 0.0]

t = np.linspace(0, 10, 101)
sol = odeint(pend, y0, t, args=(b, c))

plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')

plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#%%
# Voorbeeld minimering met Nelder Mead
def rosen_with_args(x, a, b):

    """The Rosenbrock function with additional arguments"""

    return sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + b

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = minimize(rosen_with_args, x0, method='nelder-mead', 
               args=(0.5, 1.), options={'xatol': 1e-8, 'disp': True})

print(res.x)
# %%
