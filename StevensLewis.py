#%% Stevens Lewis voorbeelde
# Begin met hierdie bladsy
# Laai al die modules

import numpy as np
import math as math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pygame 
import scipy.signal as signal
import control as ct

#%%
# Laai eie funksie modulus
from AircraftControlToolbox import *
import AircraftControl3DOF as AC3DOF
# MAAK NOG MODULE VAN AC6DOF:
import AircraftControl6DOF as AC6DOF

vliegtuig = AC3DOF.vliegtuig


# %% 3DOF voorbeelde
#---------1---------2---------3---------4---------5

#%% Voorbeeld van hoe om die 3DOF vliegtuig funksie te loop
# 
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

print(AC3DOF.f(x0, 0, vliegtuig, False, throttle, elev))



#%% Integreer vliegtuig model funksie en plot
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
sol = odeint(AC3DOF.f, x0, t, args=(vliegtuig, False, throttle, elev))

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


 
#%% Trimmer voorbeeld
# Daar is 'n handberekening vir hierdie geval in
# TransportAircraft.ods

# Druk opskrifte
print(f"{'Altitude' :^10}{'Speed' :^10}{'Throttle' :^10}{'Elevator' :^10}{'Alpha' :^10}")
print(f"{'ft' :^10}{'ft/s' :^10}{' ' :^10}{'deg' :^10}{'deg' :^10}")


# inset = [throttle, elevator [deg], theta [rad]]
inset = [0.297, -25.7, 22.1*3.14159/180]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [170, 0]
res = minimize(AC3DOF.doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")


# inset = [throttle, elevator [deg], theta [rad]]
inset = [500, 0.293, 2.46, 0.58*3.14159/180, 0]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [500, 0]
res = minimize(AC3DOF.doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")


# inset = [throttle, elevator [deg], theta [rad]]
inset = [0.204, -4.10, 5.43*3.14159/180]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [500, 30000]
res = minimize(AC3DOF.doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")



# inset = [throttle, elevator [deg], theta [rad]]
# Hierdie gee nog nie selfde antwoord as handboek nie, want dit is nie
# seker of die throttle ook opgestel is as veranderlike nie.
# MOET NOG VERFYN
inset = [0.204, -4.10, 5.43*3.14159/180]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [250, 0]
res = minimize(AC3DOF.doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-8, 'disp': False})
print('\n')
print(f"{konstant[1] :^10}{konstant[0] :^10}{res.x[0]:7.3f}{res.x[1]:8.2f}{res.x[2]*180/3.14159:8.2f}")
x0 = [konstant[0], inset[2], inset[2], 0, konstant[1], 0]
u0 = [inset[0], inset[1]]
stoorGelykVlug(x0, u0, 'Voorbeeld 3_7-6', 'Voorbeeld 3.7-6 op bladsy 173:  Transport aircraft throttle response.')


#%% Beheer 'n reghoek met die heihoek of theta wat modelleer word met 3DOF model
 
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
    xd = AC3DOF.f(x0, 0, vliegtuig, False, throttle, elev)
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
    pygame.draw.rect(win, (255, 0, 0), (100, y - vlieglengte*math.sin(x0[2]), width, height))
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

#%% Evalueer of simulasie intyds uitgevoer is

   
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

#%% Bereken die numeriese Jakobiaan

# Voorbeeld van numJakob wat die veranderlikes vooraf definieer (laai dit nie van leer af nie)
x0 = [500, 5.43*3.14159/180, 5.43*3.14159/180, 0, 30000, 0]
# u0 = [throttle, elev]
u0 = [0.204, -4.1]


A, B = numJakob(AC3DOF.vragvliegtuig, 0, u0, x0, vliegtuigfunksie = AC3DOF.f, vliegtuigmodel = vliegtuig)

drukMatriks(A)

# Voorbeeld 3.7-6:  Transport aircraft throttle response
x0, u0 = leesGelykVlug('Voorbeeld 3_7-6')
A, B = numJakob(AC3DOF.vragvliegtuig, 0, u0, x0, vliegtuigfunksie = AC3DOF.f, vliegtuigmodel = vliegtuig)

print('A:')
drukMatriks(A)
print('B:')
drukMatriks(B)

# Verwyder die laaste kolom en laaste ry om dieselfde matriks as in die voorbeeld
# op te lewer
Anuut = []
rows, cols = 4, 4
for i in range(rows):
    col = []
    for j in range(cols):
        col.append(A[i][j])
    Anuut.append(col)
drukMatriks(Anuut)

Bnuut = []
rows, cols = 4, 1
for i in range(rows):
    col = []
    for j in range(cols):
        col.append(B[i][j])
    Bnuut.append(col)
drukMatriks(Bnuut)


D = 0
C = [[1, 0, 0, 0]] # x V_T
# Find transfer function from u1 to y1
num, den = signal.ss2tf(Anuut, Bnuut, C, D, 0)
H1 = signal.TransferFunction(num, den)
print(H1)

t, y = signal.step(H1)
plt.plot(t, y)

plt.title('Step response')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.show()


# Maak control package objekte van die scipy oordragsfunksies:
H1stelsel = ct.tf(H1.num, H1.den)


print(H1stelsel)
# Bereken natuurlike frekwensie en dempingsverhouding van stelsel
odz(Anuut)



#%%


# 3DOF voorbeelde
#---------1---------2---------3---------4---------5





# %% 6DOF voorbeelde
#---------1---------2---------3---------4---------5

#%% Air data computer voorbeeld

# adc(VT, ALT)
# VT - spoed in [ft/s]
# ALT - hoogte in [ft]
AMACH, QBAR = AC6DOF.adc(500, 0)
print(AMACH)
print(QBAR)

#%% Enjin funksies

print(AC6DOF.RTAU(26))
print(AC6DOF.PDOT(50, 20))
print(AC6DOF.TGEAR(0.95))

throttlefraction = []
throttlevalue = []
# Plot throttle gearing:
for i in range(0,100):
    throttlefraction.append(i/100)
    throttlevalue.append(AC6DOF.TGEAR(throttlefraction[i]))

plt.plot(throttlefraction, throttlevalue, 'b', label='Throttle gearing')
plt.legend(loc='best')
plt.xlabel('Throttle fraction []')
plt.ylabel('Throttle [%]')
plt.grid()
plt.show()

# Voorbeeld van stukrag berekening
print(AC6DOF.THRUST(100, 0, 0.8))


#%% Plot aerodinamiese funksies


print(AC6DOF.DAMP(-10))
damplist = []
alphalist = []
for teller in range(-20, 50):
    alphalist.append(teller)
    damptemp = AC6DOF.DAMP(teller)
    damplist.append(damptemp[3])

plt.plot(alphalist, damplist, 'b', label=r'$Damp$')

print(AC6DOF.CX(0, 0))
cxlist = []
alphalist = []
for teller in range(-50, 80):
    alphalist.append(teller)
    cxlist.append(AC6DOF.CX(teller, 0))

plt.plot(alphalist, cxlist, 'b', label=r'$C_x$')

cxlist = []
alphalist = []
for teller in range(-50, 80):
    alphalist.append(teller)
    cxlist.append(AC6DOF.CX(teller, 30))

plt.plot(alphalist, cxlist, 'r', label=r'$C_x 25^\circ$')

print(AC6DOF.CZ(0, 0, 0))
czlist = []
alphalist = []
for teller in range(-50, 80):
    alphalist.append(teller)
    czlist.append(AC6DOF.CZ(teller, 0, 0))

plt.plot(alphalist, czlist, 'b', label=r'$C_z$')


print(AC6DOF.CM(0, 0))

cmlist = []
alphalist = []
for teller in range(-20, 50):
    alphalist.append(teller)
    cmlist.append(AC6DOF.CM(teller, 0))

plt.plot(alphalist, cmlist, 'b', label=r'$C_M$')

cmlist = []
alphalist = []
for teller in range(-20, 50):
    alphalist.append(teller)
    cmlist.append(AC6DOF.CM(teller, 25))

plt.plot(alphalist, cmlist, 'r', label=r'$C_M 25^\circ$')

print(AC6DOF.CL(0, 0))

cllist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    cllist.append(AC6DOF.CL(teller, 0))

plt.plot(alphalist, cllist, 'b', label=r'$C_L$')

print(AC6DOF.CN(-10, -20))

cnlist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    cnlist.append(AC6DOF.CN(teller, 0))

plt.plot(alphalist, cnlist, 'b', label=r'$C_N$')

print(AC6DOF.DLDA(-10, 0))

dldalist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dldalist.append(AC6DOF.DLDA(teller, 0))

plt.plot(alphalist, dldalist, 'b', label=r'$dLdail$')

print(AC6DOF.DLDR(-10, 0))

dldrlist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dldrlist.append(AC6DOF.DLDR(teller, 0))

plt.plot(alphalist, dldrlist, 'b', label=r'$dLdr$')

print(AC6DOF.DNDA(-10, 0))

dndalist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dndalist.append(AC6DOF.DNDA(teller, 0))

plt.plot(alphalist, dndalist, 'b', label=r'$dnda$')

print(AC6DOF.DNDR(-10, 0))

dndrlist = []
alphalist = []
for teller in range(-10, 40):
    alphalist.append(teller)
    dndrlist.append(AC6DOF.DNDR(teller, 0))

plt.plot(alphalist, dndrlist, 'b', label=r'$dndr$')


#%% Vliegtuig konstantes funksie

vliegtuig = AC6DOF.F16model()


#%% Voorbeeld van hoe om vliegtuigfunksie te roep
# f(x, t, v, THTL, ELEV, AIL, RDR, XCG)
# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, ALT, POW]
# Table 3.3-2 F-16 Model test case, page 128
x = [500.0, 0.5, -0.2, -1, 1, -1, 0.7, -0.8, 0.9, 1000, 900, 10000, 90]
xd = AC6DOF.f(x, 0, vliegtuig, 0.9, 20.0, -15.0, -20.0, 0.4)
print(xd)


#%% Integreer funksie
# Loop die funksie met Runge Kutta integrasie
# f(x, t, v, THTL, ELEV, AIL, RDR, XCG)
# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
# # Table 3.3-2 F-16 Model test case, page 128


x0 = [500.0, 0.5, -0.2, -1.0, 1.0, -1.0, 0.7, -0.8, 0.9, 1000.0, 900.0, 10000.0, 90.0]
THTL = 0.9
EL = 0.0
AIL = 0.0
RDR = 0.0
XCG = 0.4
print(AC6DOF.f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, XCG))

THTL = 0.127
EL = -1.1
AIL = 0.0
RDR = 0.0
XCG = 0.35
x0 = [350, -5.71*3.14159/180, 0, 0, -5.71*3.14159/180, 0, 0, 0, 0, 0, 0, 0, AC6DOF.TGEAR(THTL)]
print(AC6DOF.f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, XCG))
# Hierdie beginwaardes werk nog nie, ondersoek dit verder
# Die swaartepunt was verkeerd.  Dis moontlik dat hierdie stelsel marginaal of onstabiel is

THTL = 0.107
EL = 0.72
AIL = 0.0
RDR = 0.0
XCG = 0.35
x0 = [400, 4.0*3.14159/180, 0, 0, 4.0*3.14159/180, 0, 0, 0, 0, 0, 0, 0, AC6DOF.TGEAR(THTL)]
print(AC6DOF.f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, XCG))


t = np.linspace(0, 2.0, 10000)
# f(x, t, v, THTL, EL, AIL, RDR, XCG):
sol = odeint(AC6DOF.f, x0, t, args=(vliegtuig, THTL, EL, AIL, RDR, XCG))


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


#%% Euler integrasie

# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
x0 = [500.0, 0.5, -0.2, -1.0, 1.0, -1.0, 0.7, -0.8, 0.9, 1000.0, 900.0, 10000.0, 90.0]
THTL = 0.9
EL = 0.0
AIL = 0.0
RDR = 0.0
XCG = 0.4
print(AC6DOF.f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, XCG))

THTL = 0.127
EL = -1.15
AIL = 0.0
RDR = 0.0
XCG = 0.4
x0 = [350, -5.82*3.14159/180, 0, 0, -5.82*3.14159/180, 0, 0, 0, 0, 0, 0, 0, AC6DOF.TGEAR(THTL)]
print(AC6DOF.f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, XCG))

THTL = 0.107
EL = 0.72
AIL = 0.0
RDR = 0.0
XCG = 0.35
x0 = [400, 4.0*3.14159/180, 0, 0, 4.0*3.14159/180, 0, 0, 0, 0, 0, 0, 0, AC6DOF.TGEAR(THTL)]
print(AC6DOF.f(x0, 0, vliegtuig, THTL, EL, AIL, RDR, XCG))


thetaplot = [x0[4]]
alphaplot = [x0[1]]
tplot = [0]
t = 0.0

for teller in range(1, 5000):
    xd = AC6DOF.f(x0, t, vliegtuig, THTL, EL, AIL, RDR, XCG)
    xint = [i * 0.001 for i in xd]
    x0 = [x + y for x, y in zip(x0, xint)]
    
    t = t + 0.001
    thetaplot.append(x0[4])
    alphaplot.append(x0[1])
    tplot.append(t)
    
    
thetaplotdeg = [i*180/3.14159 for i in thetaplot]
alphaplotdeg = [i*180/3.14159 for i in alphaplot]
plt.plot(tplot, thetaplotdeg, 'b', label=r'$\theta (t)$')
plt.plot(tplot, alphaplotdeg, 'g', label=r'$\alpha (t)$')
plt.legend(loc='best')
plt.xlabel('Tyd [s]')
plt.ylabel(r'$\theta$ [deg]')
plt.grid()
plt.show()






#%% Beheer 'n reghoek met die heihoek of theta wat modelleer word met 6DOF model

# Beheer 'n reghoek met die heihoek of theta
from math import sin

# Beginvoorwaardes van simulasie
THTL = 0.107
trim = 0.72 # trim elevator angle
EL = trim
AIL = 0.0
RDR = 0.0
XCG = 0.35
# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
x0 = [400, 4.0*3.14159/180, 0, 0, 4.0*3.14159/180, 0, 0, 0, 0, 0, 0, 0, AC6DOF.TGEAR(THTL)]



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
text = font.render(f"{'Throttle' : ^12}{THTL:^10.3f}", True, (255,255,255))
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
    # x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
    xd = AC6DOF.f(x0, t, vliegtuig, THTL, EL, AIL, RDR, XCG)
    
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
        THTL += 0.001
        
    if keys[pygame.K_3]: 
        # increment in y co-ordinate 
        THTL -= 0.001 
            
    # completely fill the surface object 
    # with black colour 
    win.fill((0, 0, 0))
    
    # drawing object on screen which is rectangle here 
    pygame.draw.rect(win, (255, 0, 0), (100, y - vlieglengte*sin(x0[4]), width, height))
    # Teken die verwysing van die heihoek
    pygame.draw.rect(win, (0, 255, 0), (80, y, 20, 10))
    pygame.draw.rect(win, (0, 255, 0), (400, y, 20, 10))
    text = font.render(f"{'True airspeed' : ^12}{x0[0]:^10.1f}{'ft/s' :<10}", True, (255,255,255))
    win.blit(text, textSpoed)
    text = font.render(f"{'Theta' : ^12}{x0[4]*180/3.14159:^10.1f}{'deg' :<10}", True, (255,255,255))
    win.blit(text, textTheta)
    text = font.render(f"{'Throttle' : ^12}{THTL:^10.3f}", True, (255,255,255))    
    win.blit(text, textThtl)
    for joystick in joysticks.values():
        jid = joystick.get_instance_id()
        joystickPitch = joystick.get_axis(1)
    text = font.render(f"{'Joystick' : ^12}{joystickPitch:^10.3f}", True, (255,255,255))
    win.blit(text, textjoys)
    EL = trim - joystickPitch*25
    pygame.display.update()

     
 

# closes the pygame window 
pygame.quit() 


#%% Gelykvlug funksie voorbeeld

# Druk opskrifte
print(f"{'Speed' :^10}{'Throttle' :^10}{'AOA' :^10}{'Elevator' :^10}")
print(f"{'ft/s' :^10}{' ' :^10}{'deg' :^10}{'deg' :^10}")

# x = [VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, North, East, ALT, POW]
# inset = [throttle, elevator [deg], alpha [rad], aileron [deg], rudder [deg], beta [rad]]
inset = [0.123, 0.52, -5.71*3.14159/180, 0, 0, 0]
# konstant = [Spoed [ft/s], hoogte [ft]]
konstant = [350, 0, 0.35]
res = minimize(AC6DOF.doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-10, 'fatol': 1e-8, 'disp': False})
print(f"{konstant[0] :^10}{res.x[0]:7.3f}{res.x[2]*180/3.14159:8.2f}{res.x[1]:8.2f}")

spoed = [130, 140, 150, 170, 200, 260, 300, 350, 400, 440, 500, 540, 600, 640, 700, 800]

smoorklep = []

for spoedgetal in spoed:
    # Hierdie begin inset waarde is belangriker as die toleransie van minimeringsalgoritme
    inset = [0.5, 0.0, 10*3.14159/180, 0, 0, 0]
    # konstant = [Spoed [ft/s], hoogte [ft]]
    konstant = [spoedgetal, 0, 0.35]
    res = minimize(AC6DOF.doelfunksie, inset, method='nelder-mead', 
               args=(vliegtuig, konstant), options={'xatol': 1e-10, 'fatol': 1e-8, 'disp': False})
    print(f"{konstant[0] :^10}{res.x[0]:7.3f}{res.x[2]*180/3.14159:8.2f}{res.x[1]:8.2f}")
    smoorklep.append(res.x[0])

smoorklepvergelyk = [0.816, 0.736, 0.619, 0.464, 0.287, 0.148, 0.122, 0.107, 0.108, 0.113, 
                     0.137, 0.160, 0.200, 0.230, 0.282, 0.378]
plt.plot(spoed, smoorklep, 'b', label='Clean aircraft sea level')
plt.plot(spoed, smoorklepvergelyk, 'r', label='Textbook')
plt.legend(loc='best')
plt.xlabel('Spoed [ft/s]')
plt.ylabel('Smoorklep []')
plt.grid()
plt.show()

# Vergelyking met die handboek is baie goed.









# 6DOF voorbeelde
#---------1---------2---------3---------4---------5
# %%
