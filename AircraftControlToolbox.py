#%% Bevat al die algortimes benodig vir die programme in Stevens
import matplotlib.pyplot as plt
import math as math
import numpy as np


#%%
def rk4(f, dt, xx, nx, u):
    # Runge Kutta 4de orde:
    # f - funksie wat integreer moet word.
    # tt - huidige tyd
    # dt - integrasie tydstap
    # xx - toestandsveranderlike vektor
    # xd - afgeleide van xx vektor met betrekking tot tyd
    # nx - aantal veranderlikes in toestandsveranderlike vektor
    
    xd = f(xx, 0, u)

    # Inisialiseer die vektore
    x = []
    xa = []

    for m in range(0, nx):
        xa.append(xd[m]*dt)
        x.append(xx[m] + 0.5*xa[m])

    t = dt
    xd = f(x, t, u)
    for m in range(0, nx):
        q = xd[m]*dt
        x[m] = xx[m] + 0.5*q
        xa[m] = xa[m] + q + q

    xd = f(x, t, u)
    for m in range(0, nx):
        q = xd[m]*dt
        x[m] = xx[m] + q
        xa[m] = xa[m] + q + q        

    tt = dt
    xd = f(x, tt, u)
    for m in range(0, nx):
        xx[m] = xx[m] + (xa[m] + xd[m]*dt)/6.0

    return xx



#%%
# Kyk na NumerieseJakobiaan.md vir notas

def numJakob(f, t, u0, x0):
    # Funksie wat die numeriese Jakobiaan van 'n LTI (Lineer tyd onveranderlike)
    # stelsel bereken.  Die Jakobiaan word rondom 'n gestadigde bedryfspunt
    # bereken.    
    # 
    # Insette:  
    # f         - funksie van die bewegingsvergelykings
    # t         - tyd in [s] waar funksie evalueer word
    # u         - beheervektor waar rondom die Jakobiaan bereken word
    # xx        - beginwaardes waar rondom die Jakobiaan bereken word
    
    # 'n LTI stelsel is nie afhanklik van tyd nie, dus hou t = 0 arbriter
    t = 0
    
    def njvek(f, t, u0, x0, dx, ni):
        # Bepaal 'n toestandsveranderlike vektor van die funksie
        # ni is die nommer van die veranderlike wat versteur word
        # dx is die waarde waarmee die veranderlike versteur word

        xd0 = f(x0, t, u0) # vektor by gestadigde toestand
        x1 = [item for item in x0] # Inisialiseer die tweede vektor by versteurde toestand
        
        x1[ni] += dx
        xd1 = f(x1, t, u0)

        njvek = []
        # Bereken die gradient vir elke veranderlike
        for xd0el, xd1el in zip(xd0, xd1):
            njvek.append((xd1el - xd0el)/dx)
        
        return njvek

    def njvekkonv(f, t, u0, x0, ni):
        # Konvergeer die Jakobiaan vektor totdat minder as 0.1% 
        # verandering in die grootste verandering gebeur het.
            
        dx = 1.0
        tolcheck = 1

        while tolcheck > 0.0001:
            njvek0 = njvek(f, t, u0, x0, dx, ni)
            dx = dx/2
            njvek1 = njvek(f, t, u0, x0, dx, ni)

            tolerance = []

            for el0, el1 in zip(njvek0, njvek1):
                tolerance.append(abs(el1 - el0))

            tolcheck = max(tolerance)
            #print(f"{'Tolerance '}{tolcheck}{' dx '}{dx}")

        njvekkonv = njvek1

        return njvekkonv
    
    Jakobiaant = []
    
    for ni in range(0, len(x0)):
        Jakobiaant.append(njvekkonv(f, t, u0, x0, ni))

    # star operator will first
    # unpack the values of 2D list
    # and then zip function will 
    # pack them again in opposite manner
    # Transponeer die matriks
    Jakobiaan = list(map(list, zip(*Jakobiaant)))

    return Jakobiaan


f = pendulum
u0 = [0]
x0 = [0, 0]
print('Numeriese berekening van Jakobiaan')
print(numJakob(f, 0, u0, x0))

regtewaarde = [[0, 1], [-9.81/3, 0]]
print(f"{'Analitiese waarde van Jakobiaan '}{regtewaarde}")
print('Hier is die eiewaardes en eievektore')
eiewaardes, eievektore = np.linalg.eig(np.array(regtewaarde))
print(eiewaardes)

#%%

def puntmassa(x, t, u):
    # Bereken die posisie van 'n puntmassa, gegewe die beginsnelheid
    # en die versnelling.  Die versnelling in hierdie geval is 
    # swaartekrag op aarde, naamlik 9.81m/s^2
    #
    # s = ut + 0.5at^2
    # v = u + at
    # 
    a = 9.81 # Gravitasieversnelling [m/s^2]
    xd = [x[1], 9.81]
    return xd

# voorbeeld van die puntmassa se toestandsveranderlike vektor:
print(puntmassa([10, 10], 0, []))

dt = 0.01 # sekondes
tydvektor = [i*dt for i in range(0, 1001)]
afstand = [0]

# Beginwaardes
x0 = [0, 0]
xx =[]

for tyd in tydvektor:
    xx = rk4(puntmassa, dt, x0, len(x0), [])
    afstand.append(xx[0])
    

plt.plot(tydvektor, afstand[0:-1], 'b', label=r's')

plt.legend(loc='best')
plt.xlabel('t [s]')
plt.ylabel('afstand [m]')
plt.grid()
plt.show()

print(f"{'Antwoord moet wees '}{0.5*9.81*(0.01*1000)**2}")
print(f"{'Antwoord moet is '}{afstand[-2]:3.8f}")


#%%
# Nie-lineere toestandsveranderlike stelsel wat 'n pendulum beskryf
# Hierdie stelsel word gebruik om die numeriese Jakobiaan funksie te ontfout

def pendulum(x, t, u):
    l = 3 # [m]
    m = 0.1 # [kg]

    xd = [0, 0]

    xd[0] = x[1]
    xd[1] = -(9.81/l)*math.sin(x[0]) + (1/(l*m)*u[0]**2)

    return xd


# voorbeeld van die pendulum se toestandsveranderlike vektor:
# Rondom die ekwilibrium punt
print(pendulum([0, 0], 0, [0]))

dt = 0.01 # sekondes
tydvektor = [i*dt for i in range(0, 1001)]
afstand = [0]

# Beginwaardes
x0 = [0.05, 0]
xx =[]
u = [0.5]

for tyd in tydvektor:
    xx = rk4(pendulum, dt, x0, len(x0), u)
    afstand.append(xx[0]*180/3.14159)
    

plt.plot(tydvektor, afstand[0:-1], 'b', label=r'$\theta$')

plt.legend(loc='best')
plt.xlabel('t [s]')
plt.ylabel('Hoek [grade]')
plt.grid()
plt.show()



#%%
# Hoe om 'n funksie in 'n ander funksie te roep:
# Hierdie voorbeeld word benodig om byvoorbeeld die 
# vliegtuigmodel funksie te stuur na die simpleks
# algoritme of die Jakobiaan algoritme
#
# Python program to illustrate functions 
# can be passed as arguments to other functions 
def shout(text): 
    return text.upper() 

def whisper(text): 
    return text.lower() 

def greet(func): 
    # storing the function in a variable 
    greeting = func("Hi, I am created by a function passed as an argument.") 
    print(greeting)

greet(shout) 
greet(whisper)
# %%
