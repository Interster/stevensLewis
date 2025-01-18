#%% Voorbeelde van verskillende pakette wat benodig word vir Stevens Lewis

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

#%%
# Verander toestandsveranderlike formulasie na oordragsfunksie
import control.matlab as ct


A = [[-1, -2], [3, -4]]
B = [[5], [6]]
C = [[7, 8]]
D = [[9]]

sys1 = ct.ss2tf(A, B, C, D)
print(sys1)
# %%
# MIMO stelsels
# Hier is 'n verduideliking van MIMO stelsels
# https://www.youtube.com/watch?v=ascHxHLpMHg
# Oordragsfunksie met Python:
# https://www.youtube.com/watch?v=9XgvL05KPE4

# vir MIMO om te werk moet slycot installeer word.
# dit steun op gfortran en OpenBLAS
# moes dus OpenBLAS installeer:
# sudo apt-get install -y libopenblas-dev

import control as ct


A = [[0, 1], [-1, -3]]
B = [[0, 0], [2, 4]]
C = [[1, 0]]
D = [[0, 0]]

sys1 = ct.ss2tf(A, B, C, D)
print(sys1)
# %%
# Gebruik die scipy module vir die MIMO gedeelte want slycot is problematies
# om te vertaal want dit is moeilik om f2py te kry vir python3 waarop dit berus
# Dit kom van hierdie youtube video:
# https://www.youtube.com/watch?v=Hr8Ssxk59Ps

import scipy.signal as signal
import matplotlib.pyplot as plt

A = [[0, 1], [-1, -3]]
B = [[0, 0], [2, 4]]
C = [[1, 0]]
D = [[0, 0]]

# Find transfer function from u1 to y
num, den = signal.ss2tf(A, B, C, D, 0)
H1 = signal.TransferFunction(num, den)
print(H1)

# Find transfer function from u2 to y
num, den = signal.ss2tf(A, B, C, D, 1)
H2 = signal.TransferFunction(num, den)
print(H1)

t, y = signal.step(H1)
plt.plot(t, y)
t, y = signal.step(H2)
plt.plot(t, y)

plt.title('Step response')
plt.xlabel('t')
plt.ylabel('y')
plt. legend(["H1", "H2"])
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
# Nou vir MIMO voorbeeld;
# Dit kom van hierdie youtube video:
# https://www.youtube.com/watch?v=Hr8Ssxk59Ps

import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ct

A = [[0, 1], [-1, -3]]
B = [[0, 0], [2, 4]]
D = [[0, 0]]


A = [[0, 1], [0, -1]]
B = [[0], [1]]
D = 0


C = [[1, 0]]
# Find transfer function from u1 to y1
num, den = signal.ss2tf(A, B, C, D, 0)
H1 = signal.TransferFunction(num, den)
print(H1)

# Find transfer function from u2 to y1
num, den = signal.ss2tf(A, B, C, D, 1)
H2 = signal.TransferFunction(num, den)
print(H2)

# Die MIMO word gedoen met hierdie C output matriks
C = [[0, 1]]
# Find transfer function from u1 to y2
num, den = signal.ss2tf(A, B, C, D, 0)
H3 = signal.TransferFunction(num, den)
print(H3)

# Find transfer function from u2 to y2
num, den = signal.ss2tf(A, B, C, D, 1)
H4 = signal.TransferFunction(num, den)
print(H4)


t, y = signal.step(H1)
plt.plot(t, y)
t, y = signal.step(H2)
plt.plot(t, y)
t, y = signal.step(H3)
plt.plot(t, y)
t, y = signal.step(H4)
plt.plot(t, y)

plt.title('Step response')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(["H1", "H2", "H3", "H4"])
plt.grid()
plt.show()


# Maak control package objekte van die scipy oordragsfunksies:
H1stelsel = ct.tf(H1.num, H1.den)
H2stelsel = ct.tf(H2.num, H2.den)
H3stelsel = ct.tf(H3.num, H3.den)
H4stelsel = ct.tf(H4.num, H4.den)

print(H1stelsel)
print(H2stelsel)
print(H3stelsel)
print(H4stelsel)

#%% Funksies met varierende aantal insette

# Definieer alle konstantes met 'n dictionary
tuig = {'S' : 2170.0,
        'CBAR' : 17.5,
        'AM' : 5e3
        }

def voorbeeld1(t, u, x):
    print(u*x)


def voorbeeld2(t, u, x, vt):
    print(u*x)
    print(vt['S'])

def evalueerFunksie(f, t, u, x, **kwargs):
    if kwargs.__len__() > 0:
        f(t, u, x, kwargs['vliegtuigmodel'])
    else:
        f(t, u, x)


t = 0
u = 3
x = 4


evalueerFunksie(voorbeeld1, t, u, x)
evalueerFunksie(voorbeeld2, t, u, x, vliegtuigmodel = tuig)



#%% Nou vir MIMO voorbeeld;  Uit Stevens & Lewis:
# Bl. 174:  Transport aircraft throttle response

import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ct
import numpy as np

# V_T, alpha, theta, q
A = [[-1.6096e-2, 1.8832e1, -3.217e1, 0.0], 
     [-1.0189e-3, -6.3537e-1, 0.0, 1.0],
     [0.0, 0.0, 0.0, 1.0],
     [1.0744e-4, -7.7544e-1, 0.0, -5.2977e-1]]
# delta_{throttle}
B = [[9.9679], 
     [-6.513e-3], 
     [0.0], 
     [2.5575e-2]]
D = 0


C = [[1, 0, 0, 0]] # x V_T
# Find transfer function from u1 to y1
num, den = signal.ss2tf(A, B, C, D, 0)
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

# Bereken eiewaardes en eievektore van stelsel
eiewaardes, eievektore = np.linalg.eig(np.array(A))

#%% Vergelyk dit met die antwoord op bl. 174 en gebruik sympy
import sympy
sympy.init_printing()

# Maak 'n Laplace veranderlike s
# Die Laplace metodes kom uit hierdie webwerf dokumentasie:
# https://dynamics-and-control.readthedocs.io/en/latest/index.html
t, s = sympy.symbols('t, s')

# oordragsfunksie 
VTdeltath = (9.968*(s - 0.0601)*(s + 0.6065 - 0.8811j)*(s + 0.6065 + 0.8811j))/ \
((s + 2.277e-4 + 0.1567j)*(s + 2.277e-4 - 0.1567j)*(s + 0.5904 + 0.8811j)*(s + 0.5904 - 0.8811j))

# Gee selfde antwoord as die beheerstelsel module hierbo
VTdeltath.simplify()

# Die inverse Laplace is as volg:
F = VTdeltath.simplify()
ft = sympy.inverse_laplace_transform(F, s, t)

ft.simplify()

#%% Verskillende maniere om komplekse getalle te definieer

# Volgende metodes gee selfde resultate
z1 = complex(0.6065, 0.8811)
z2 = complex(0.6065, -0.8811)

z1 = 0.6065 + 0.8811j

print(z1*z2)

#%% Gebruik sympy
import sympy
sympy.init_printing()

x = sympy.symbols('x')
sympy.init_printing()

sympy.Integral(sympy.sqrt(1/x), x)

# Substitusie
expr = sympy.cos(x) + 1
nuwe = expr.subs(x, 2*x)

# Evalueer 'n funksie
# Substitueer eers pi in "nuwe" in en evalueer dan
evalueer = nuwe.subs(x, 3.14159)
print(evalueer.evalf())

#%%

        

