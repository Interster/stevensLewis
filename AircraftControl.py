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


#%% 
# Voorbeeld van hoe om die vliegtuig funksie te loop
# f(x, t, v, land, THTL, ELEV)
# x = [VT, ALPHA, THETA, Q, H, Distance]
x0 = [170, 22.1*3.14159/180, 22.1*3.14159/180, 0, 0, 0]
throttle = 0.297
elev = -25.7

x0 = [500, 0.58*3.14159/180, 0.58*3.14159/180, 0, 0, 0]
throttle = 0.293
elev = 2.46

x0 = [500, 5.43*3.14159/180, 5.43*3.14159/180, 0, 30000, 0]
throttle = 0.204
elev = -4.1

print(f(x0, 0, vliegtuig, False, throttle, elev))

#%% Integreer funksie
# Loop die funksie met Runge Kutta integrasie
#x0 = [spoed, alpha, theta, heiversnelling, 
# opwaartse spoed, horizontale spoed]
x0 = [170, 22.1*3.14159/180, 22.1*3.14159/180, 0, 0, 0]
throttle = 0.297
elev = -25.7

x0 = [500, 0.58*3.14159/180, 0.58*3.14159/180, 0, 0, 0]
throttle = 0.293
elev = 2.46

x0 = [500, 5.43*3.14159/180, 5.43*3.14159/180, 0, 30000, 0]
throttle = 0.204
elev = -4.1

t = np.linspace(0, 10, 101)
sol = odeint(f, x0, t, args=(vliegtuig, False, throttle, elev))

plt.plot(t, sol[:, 2]*180/3.14159, 'b', label=r'$\theta (t)$')
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
# Daar is 'n handberekening vir hierdie geval in
# TransportAircraft.ods

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


# inset = [throttle, elevator [deg], theta [rad]]
inset = [500, 0.293, 2.46, 0.58*3.14159/180, 0]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [500, 0]
res = minimize(doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")


# inset = [throttle, elevator [deg], theta [rad]]
inset = [0.204, -4.10, 5.43*3.14159/180]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [500, 30000]
res = minimize(doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")


# %%
# Plot simulasie tyd en werklike tyd saam 
# Kyk of daar 'n verskil is
# Dit blyk dat daar nie probleem is om intyds te loop nie
# Processor	11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz   1.38 GHz
# Installed RAM	32.0 GB (31.7 GB usable)


plt.plot(ttemp, 'b', label='Simulasie tyd')
plt.plot(ttick, 'g', label=r'Regte tyd')

plt.legend(loc='best')
plt.ylabel('tyd [milisekondes]')
plt.grid()
plt.show()


# %%
# Beheer 'n reghoek met die heihoek of theta
from math import sin

# Beginvoorwaardes van simulasie
# x = [VT, ALPHA, THETA, Q, H, Distance]
x0 = [500, 0.58*3.14159/180, 0.58*3.14159/180, 0, 0, 0]
throttle = 0.293
trim = 2.46 # trim elevator angle
elev = trim
# Integration time step in milliseconds
dt = 10
# Lengte van vliegtuig projeksie
vlieglengte = 200

# import pygame module in this program 
import pygame 

# activate the pygame library . 
# initiate pygame and give permission 
# to use pygame's functionality. 
pygame.init() 

# create the display surface object 
# of specific dimension..e(500, 500). 
win = pygame.display.set_mode((500, 500)) 

# set the pygame window name 
pygame.display.set_caption("Vliegtuig heihoek") 

# object current co-ordinates 
x = 100
y = 250

# dimensions of the object 
width = 300
height = 5

# velocity / speed of change of control elevator
vel = 0.1

# Indicates pygame is running 
run = True

# create a font object.
# 1st parameter is the font file
# which is present in pygame.
# 2nd parameter is size of the font
font = pygame.font.Font('freesansbold.ttf', 16)

# create a text surface object,
# on which text is drawn on it.
text = font.render(f"{'True airspeed' : ^12}{x0[0]:^10.1f}{'ft/s' :<10}", True, (255,255,255)) 
# create a rectangular object for the
# text surface object
textSpoed = text.get_rect()
# set the center of the rectangular object.
textSpoed.center = (150, 150)
text = font.render(f"{'Theta' : ^12}{x0[2]*180/3.14159:^10.1f}{'deg' :<10}", True, (255,255,255))
textTheta = text.get_rect()
# set the center of the rectangular object.
textTheta.center = (150, 180)
text = font.render(f"{'Throttle' : ^12}{throttle:^10.3f}", True, (255,255,255))
textThtl = text.get_rect()
# set the center of the rectangular object.
textThtl.center = (150, 210)

joystickPitch = 0.0
text = font.render(f"{'Joystick pitch' : ^12}{joystickPitch:^10.3f}", True, (255,255,255))
textjoys = text.get_rect()
# set the center of the rectangular object.
textjoys.center = (150, 350)



# Tydelike tydstempel
ttemp = [0]
ttick = [0]

# This dict can be left as-is, since pygame will generate a
# pygame.JOYDEVICEADDED event for every joystick connected
# at the start of the program.
joysticks = {}

for joystick in joysticks.values():
    jid = joystick.get_instance_id()

# infinite loop 
while run:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True  # Flag that we are done so we exit this loop.

            if event.type == pygame.JOYBUTTONDOWN:
                print("Joystick button pressed.")
                if event.button == 0:
                    joystick = joysticks[event.instance_id]
                    if joystick.rumble(0, 0.7, 500):
                        print(f"Rumble effect played on joystick {event.instance_id}")

            if event.type == pygame.JOYBUTTONUP:
                print("Joystick button released.")

            # Handle hotplugging
            if event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for every
                # joystick, filling up the list without needing to create them manually.
                joy = pygame.joystick.Joystick(event.device_index)
                joysticks[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connencted")

            if event.type == pygame.JOYDEVICEREMOVED:
                del joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")
    # f(x, t, v, land, THTL, ELEV)
    xd = f(x0, 0, vliegtuig, False, throttle, elev)
    xint = [i * (dt/1000) for i in xd]
    x0 = [x + y for x, y in zip(x0, xint)]
    # creates time delay of 10ms 
    pygame.time.delay(dt)
    # Neem die tyd op om te kyk of die simulasie intyds is
    ttick = ttick + [pygame.time.get_ticks()]
    ttemp = ttemp + [ttemp[-1] + dt]
    # iterate over the list of Event objects 
    # that was returned by pygame.event.get() method. 
    for event in pygame.event.get(): 
        # if event object type is QUIT 
        # then quitting the pygame 
        # and program both. 
       if event.type == pygame.QUIT: 
            # it will make exit the while loop 
            run = False
    # stores keys pressed 
    keys = pygame.key.get_pressed()

    # if left arrow key is pressed 
    if keys[pygame.K_LEFT] and x>0:
        # decrement in x co-ordinate
        x -= vel 
    # if left arrow key is pressed 
    if keys[pygame.K_RIGHT] and x<500-width: 
        
        # increment in x co-ordinate 
        x += vel 
        
    # if left arrow key is pressed 
    if keys[pygame.K_UP] and y>0: 
        
        # decrement in y co-ordinate 
        trim += vel 
        
    # if left arrow key is pressed 
    if keys[pygame.K_DOWN] and y<500-height: 
        # increment in y co-ordinate 
        trim -= vel 
        
    # if left arrow key is pressed 
    if keys[pygame.K_9]: 
        # increment in y co-ordinate 
        throttle += 0.001
        
    if keys[pygame.K_3]: 
        # increment in y co-ordinate 
        throttle -= 0.001 
            
    # completely fill the surface object 
    # with black colour 
    win.fill((0, 0, 0))
    
    # drawing object on screen which is rectangle here 
    pygame.draw.rect(win, (255, 0, 0), (100, y - vlieglengte*sin(x0[2]), width, height))
    # Teken die verwysing van die heihoek
    pygame.draw.rect(win, (0, 255, 0), (80, y, 20, 10))
    pygame.draw.rect(win, (0, 255, 0), (400, y, 20, 10))
    text = font.render(f"{'True airspeed' : ^12}{x0[0]:^10.1f}{'ft/s' :<10}", True, (255,255,255))
    win.blit(text, textSpoed)
    text = font.render(f"{'Theta' : ^12}{x0[2]*180/3.14159:^10.1f}{'deg' :<10}", True, (255,255,255))
    win.blit(text, textTheta)
    text = font.render(f"{'Throttle' : ^12}{throttle:^10.3f}", True, (255,255,255))    
    win.blit(text, textThtl)
    for joystick in joysticks.values():
        jid = joystick.get_instance_id()
        joystickPitch = joystick.get_axis(1)
    text = font.render(f"{'Joystick' : ^12}{joystickPitch:^10.3f}", True, (255,255,255))
    win.blit(text, textjoys)
    elev = trim - joystickPitch*25
    pygame.display.update()

     
 

# closes the pygame window 
pygame.quit() 


    





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
# Voorbeeld van pygame wat 'n reghoeg beheer
# Die reghoe wys die toestand van die heihoek

# import pygame module in this program 
import pygame 

# activate the pygame library . 
# initiate pygame and give permission 
# to use pygame's functionality. 
pygame.init() 

# create the display surface object 
# of specific dimension..e(500, 500). 
win = pygame.display.set_mode((500, 500)) 

# set the pygame window name 
pygame.display.set_caption("Moving rectangle") 

# object current co-ordinates 
x = 200
y = 200

# dimensions of the object 
width = 300
height = 5

# velocity / speed of movement 
vel = 10

# Indicates pygame is running 
run = True

# infinite loop 
while run: 
    # creates time delay of 10ms 
    pygame.time.delay(100) 
    
    # iterate over the list of Event objects 
    # that was returned by pygame.event.get() method. 
    for event in pygame.event.get(): 
        
        # if event object type is QUIT 
        # then quitting the pygame 
        # and program both. 
        if event.type == pygame.QUIT: 
            
            # it will make exit the while loop 
            run = False
    # stores keys pressed 
    keys = pygame.key.get_pressed() 
    
    # if left arrow key is pressed 
    if keys[pygame.K_LEFT] and x>0: 
        
        # decrement in x co-ordinate 
        x -= vel 
        
    # if left arrow key is pressed 
    if keys[pygame.K_RIGHT] and x<500-width: 
        
        # increment in x co-ordinate 
        x += vel 
        
    # if left arrow key is pressed 
    if keys[pygame.K_UP] and y>0: 
        
        # decrement in y co-ordinate 
        y -= vel 
        
    # if left arrow key is pressed 
    if keys[pygame.K_DOWN] and y<500-height: 
        # increment in y co-ordinate 
        y += vel 
        
            
    # completely fill the surface object 
    # with black colour 
    win.fill((0, 0, 0)) 
    
    # drawing object on screen which is rectangle here 
    pygame.draw.rect(win, (255, 0, 0), (x, y, width, height)) 
    
    # it refreshes the window 
    pygame.display.update() 

# closes the pygame window 
pygame.quit() 

#%%
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.key(0)

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = jnp.array([True, True, False, True])

# Training loss is the negative log-likelihood of the training examples.
def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))

# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,))
b = random.normal(b_key, ())
#%%
# Bereken die numeriese Jakobiaan vir linearisasie met die
# volgende funksies:
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev
# Hierdie bereken die Jakobiaan van 'n funksie wat
# opgestel is met jax.
# Moet nou 'n funksie van die vliegtuigmodel opstel
# en dit in 'n formaat sit wat jacfwd aanvaar.
# Dan kan jakobiaan bereken word.


from jax import jacfwd, jacrev

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

J = jacfwd(f)(W)
print("jacfwd result, with shape", J.shape)
print(J)

J = jacrev(f)(W)
print("jacrev result, with shape", J.shape)
print(J)
# %%
# Hier is 'n eenvoudiger voorbeeld om jacfwd te gebruik:
# Maak die vliegtuigfunksie dat dit 'n jax numpy array uitgee
# en dan kan die Jakobiaan bereken word.
# https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html

import jax
import jax.numpy as jnp

def f(x):
     return jnp.asarray(
           [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])

print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))

# %%
