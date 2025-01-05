#%% Bevat al die algortimes benodig vir die programme in Stevens
import matplotlib.pyplot as plt

#%%
def rk4(f, dt, xx, nx):
    # Runge Kutta 4de orde:
    # f - funksie wat integreer moet word.
    # tt - huidige tyd
    # dt - integrasie tydstap
    # xx - toestandsveranderlike vektor
    # xd - afgeleide van xx vektor met betrekking tot tyd
    # nx - aantal veranderlikes in toestandsveranderlike vektor
    
    xd = f(xx, 0)

    # Inisialiseer die vektore
    x = []
    xa = []

    for m in range(0, nx):
        xa.append(xd[m]*dt)
        x.append(xx[m] + 0.5*xa[m])

    t = dt
    xd = f(x, t)
    for m in range(0, nx):
        q = xd[m]*dt
        x[m] = xx[m] + 0.5*q
        xa[m] = xa[m] + q + q

    xd = f(x, t)
    for m in range(0, nx):
        q = xd[m]*dt
        x[m] = xx[m] + q
        xa[m] = xa[m] + q + q        

    tt = dt
    xd = f(x, tt)
    for m in range(0, nx):
        xx[m] = xx[m] + (xa[m] + xd[m]*dt)/6.0

    return xx







#%%

def puntmassa(x, t):
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
print(puntmassa([10, 10], 0))

dt = 0.01 # sekondes
tydvektor = [i*dt for i in range(0, 1001)]
afstand = [0]

# Beginwaardes
x0 = [0, 0]
xx =[]

for tyd in tydvektor:
    xx = rk4(puntmassa, dt, x0, len(x0))
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
