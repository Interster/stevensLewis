#%% Stevens Lewis voorbeelde
# Begin met hierdie bladsy
# Laai al die modules

import numpy as np
import math as math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pygame 

#%%
# Laai eie funksie modulus
from AircraftControlToolbox import *
import AircraftControl3DOF as AC3DOF
# MAAK NOG MODULE VAN AC6DOF:
#import AircraftControl6DOF as AC6DOF

vliegtuig = AC3DOF.vliegtuig


# %% 3DOF voorbeelde
#---------1---------2---------3---------4---------5

#%% Voorbeeld van hoe om die vliegtuig funksie te loop
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


x0 = [500, 5.43*3.14159/180, 5.43*3.14159/180, 0, 30000, 0]
# u0 = [throttle, elev]
u0 = [0.204, -4.1]


A, B = numJakob(AC3DOF.vragvliegtuig, 0, u0, x0, vliegtuigfunksie = AC3DOF.f, vliegtuigmodel = vliegtuig)

print(A)

#%%


# 3DOF voorbeelde
#---------1---------2---------3---------4---------5





# %% 6DOF voorbeelde
#---------1---------2---------3---------4---------5




# 6DOF voorbeelde
#---------1---------2---------3---------4---------5