#%% Bevat al die algortimes benodig vir die programme in Stevens
import matplotlib.pyplot as plt
import math as math
import numpy as np
import pygame 

#%% Runge-Kutta integrasie algoritme
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

#%% Pendulum funksie
# Nie-lineere toestandsveranderlike stelsel wat 'n pendulum beskryf
# Hierdie stelsel word gebruik om die numeriese Jakobiaan funksie te ontfout

def pendulum(x, t, u):
    l = 3 # [m]
    m = 0.1 # [kg]

    xd = [0, 0]

    xd[0] = x[1]
    xd[1] = -(9.81/l)*math.sin(x[0]) + (1/(l*m)*u[0]**2)

    return xd


def voorbeeldIntegrasiePendulum():
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


#%% Puntmassa funksie

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


def voorbeeldIntegrasiePuntMassa():
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



#%% Numeriese Jakobiaan funksie
# Kyk na NumerieseJakobiaan.md vir notas

def numJakob(f, t, u0, x0, **kwargs):
    # Funksie wat die numeriese Jakobiaan van 'n LTI (Lineer tyd onveranderlike)
    # stelsel bereken.  Die Jakobiaan word rondom 'n gestadigde bedryfspunt
    # bereken.    
    # 
    # Insette:  
    # f         - funksie van die bewegingsvergelykings OF die omhulselfunksie 
    #             van die bewegingsvergelykings
    # t         - tyd in [s] waar funksie evalueer word
    # u         - beheervektor waar rondom die Jakobiaan bereken word
    # xx        - beginwaardes waar rondom die Jakobiaan bereken word
    #
    # Indien daar ekstra insette is:
    # kwargs['vliegtuigfunksie']        - vliegtuigfunksie wat bewegingsvergelykings bevat
    # kwargs['vliegtuigmodel']          - vliegtuigmodel in dictionary formaat
    
    # 'n LTI stelsel is nie afhanklik van tyd nie, dus hou t = 0 arbriter
    t = 0
    
    def njvek(f, t, u0, x0, dx, ni, tipe, **kwargs):
        # Bereken numeriese Jakobiaan vektor
        # Bepaal 'n toestandsveranderlike vektor van die funksie
        # ni is die nommer van die veranderlike wat versteur word
        # dx is die waarde waarmee die veranderlike versteur word
        # tipe is 'A' of 'B' en dui die stelsel of inset Jakobiaan aan

        if kwargs.__len__() > 0:
            xd0 = f(x0, t, u0, 
                    kwargs['vliegtuigfunksie'], 
                    kwargs['vliegtuigmodel'])
        else:
            xd0 = f(x0, t, u0) # vektor by gestadigde toestand
        
        x1 = [item for item in x0] # Inisialiseer die tweede vektor by versteurde toestand
        u1 = [item for item in u0] # Inisialiseer die tweede vektor by versteurde toestand
        
        if tipe == 'A':
            x1[ni] += dx
            if kwargs.__len__() > 0:
                xd1 = f(x1, t, u0, 
                    kwargs['vliegtuigfunksie'], 
                    kwargs['vliegtuigmodel'])
            else:
                xd1 = f(x1, t, u0)
        elif tipe == 'B':
            u1[ni] += dx
            if kwargs.__len__() > 0:
                xd1 = f(x0, t, u1, 
                    kwargs['vliegtuigfunksie'], 
                    kwargs['vliegtuigmodel'])
            else:
                xd1 = f(x0, t, u1)

        njvek = []
        # Bereken die gradient vir elke veranderlike
        for xd0el, xd1el in zip(xd0, xd1):
            njvek.append((xd1el - xd0el)/dx)
        
        return njvek

    def njvekkonv(f, t, u0, x0, ni, tipe, **kwargs):
        # Konvergeer die Jakobiaan vektor totdat minder as 0.1% 
        # verandering in die grootste verandering gebeur het.
            
        dx = 1.0
        tolcheck = 1

        while tolcheck > 0.0001:
            if kwargs.__len__() > 0:
                njvek0 = njvek(f, t, u0, x0, dx, ni, tipe, 
                    vliegtuigfunksie = kwargs['vliegtuigfunksie'], 
                    vliegtuigmodel = kwargs['vliegtuigmodel'])
            else:
                njvek0 = njvek(f, t, u0, x0, dx, ni, tipe)
            dx = dx/2
            if kwargs.__len__() > 0:
                njvek1 = njvek(f, t, u0, x0, dx, ni, tipe, 
                    vliegtuigfunksie = kwargs['vliegtuigfunksie'], 
                    vliegtuigmodel = kwargs['vliegtuigmodel'])
            else:
                njvek1 = njvek(f, t, u0, x0, dx, ni, tipe)

            tolerance = []

            for el0, el1 in zip(njvek0, njvek1):
                tolerance.append(abs(el1 - el0))

            tolcheck = max(tolerance)
            #print(f"{'Tolerance '}{tolcheck}{' dx '}{dx}")

        njvekkonv = njvek1

        return njvekkonv
    
    # Bereken stelsel Jakobiaan
    Jakobiaant = []
    
    for ni in range(0, len(x0)):
        if kwargs.__len__() > 0:
            Jakobiaant.append(njvekkonv(f, t, u0, x0, ni, 'A', 
                                        vliegtuigfunksie = kwargs['vliegtuigfunksie'], 
                                        vliegtuigmodel = kwargs['vliegtuigmodel']))
        else:
            Jakobiaant.append(njvekkonv(f, t, u0, x0, ni, 'A'))

    # star operator will first
    # unpack the values of 2D list
    # and then zip function will 
    # pack them again in opposite manner
    # Transponeer die matriks
    JakobiaanA = list(map(list, zip(*Jakobiaant)))
    
    # Bereken inset Jakobiaan
    Jakobiaant = []

    for ni in range(0, len(u0)):
        if kwargs.__len__() > 0:
            Jakobiaant.append(njvekkonv(f, t, u0, x0, ni, 'B', 
                                        vliegtuigfunksie = kwargs['vliegtuigfunksie'], 
                                        vliegtuigmodel = kwargs['vliegtuigmodel']))
        else:
            Jakobiaant.append(njvekkonv(f, t, u0, x0, ni, 'B'))

    # Transponeer die matriks
    JakobiaanB = list(map(list, zip(*Jakobiaant)))

    return JakobiaanA, JakobiaanB


def voorbeeldJakobiaan():
    f = pendulum
    u0 = [(0.5)**0.5]
    x0 = [0, 0]
    print('Numeriese berekening van Jakobiaan')
    print(numJakob(f, 0, u0, x0))

    regtewaarde = [[0, 1], [-9.81/3, 0]]
    print(f"{'Analitiese waarde van Jakobiaan A'}{regtewaarde}")
    print('Hier is die eiewaardes en eievektore')
    eiewaardes, eievektore = np.linalg.eig(np.array(regtewaarde))
    print(eiewaardes)
    regtewaarde = [[0], [(2/(0.1*3)) *(0.5)**0.5]]
    print(f"{'Analitiese waarde van Jakobiaan B vir F = 0.5'}{regtewaarde}")

#%% Natuurlike frekwensie en dempingsverhouding funksie

def odz(A):
    # Funksie wat omega gedemp en zeta bereken van die A matriks

    eiewaardes, eievektore = np.linalg.eig(np.array(A))
    for i in eiewaardes:
        print('\n')
        print(f'{'Eiewaarde '}{i}')

        a = i.real
        b = i.imag
        zeta = (1/((b/a)**2 + 1))**0.5
        omegan = -a/zeta

        if b != 0 and b > 0:
            print(f'{'Damped natural frequency '}{b:.2f}{' rad/s'}')
            print(f'{'Damped natural frequency '}{b/2/3.14159:.2f}{' Hz'}')
            print(f'{'Period of damped natural frequency '}{(2*3.14159)/b:.2f}{' s'}')
            print(f'{'Damping ratio '}{zeta:.3f}')
        elif b == 0:
            print(f'{'Time constant '}{-1/a:.2f}{' s'}')


def voorbeeldOmegaZeta():
    # Longitudinale modes, Stevens & Lewis bl. 164
    # Example 3.7-3 F-16 longitudinal modes
    import numpy as np

    # V_T, alpha, theta, q
    A = [[-2.0244e-2, 7.8761, -3.2169e1, -6.502e-1 ], 
        [-2.5373e-4, -1.0189, 0.0, 9.0484e-1],
        [0.0, 0.0, 0.0, 1.0],
        [7.9472e-11, -2.4982, 0.0, -1.3861]]
            
    print('Example 3.7-3 F-16 longitudinal modes')
    odz(A)

    # Longitudinale modes, Stevens & Lewis bl. 165
    # Example 3.7-4 F-16 lateral-directional modes
    import numpy as np

    # V_T, alpha, theta, q
    A = [[-3.22e-1, 6.4032e-2, 3.8904e-2, -9.9156e-1], 
        [0.0, 0.0, 1.0, 3.9385e-2],
        [-3.0919e1, 0.0, -3.673, 6.7425e-1],
        [9.4724, 0.0, -2.6358e-2, -4.9849e-1]]

    print('Example 3.7-4 F-16 lateral-directional modes')
    odz(A)


#%% Skryf en lees gelykvlug waardes

def stoorGelykVlug(x0, u0, leernaam, beskrywing):
    # Stoor die gelykvlug waardes in 'n *.csv leer
    #
    # Insette:
    # x0        - Beginwaardes
    # u0        - Beheerinsetwaardes by gelykvlug
    # leernaam  - Leernaam van die *.csv leer waar die data gestoor word
    # beskrywing- Beskryf die gelykvlug toestand bv.: V = 500ft/s gelykvlug
    
    def lysSkryf(lys, lysnaam, file):
        # Maak 'n string van die veranderlike, skryf na die leer en sit komma by
        # Veranderlike naam skryf
        file.write(f'{lysnaam}{', '}')

        stringlys = []
        for i in lys:
            stringlys.append(str(i))
        
        file.write(','.join(stringlys)) 
        file.write('\n')

    
    file = open(f'{leernaam}{'.csv'}', "w") 
    file.write(f'{beskrywing}{'\n'}')
    file.write('3DOF x0 = [VT, ALPHA, THETA, Q, H, Distance]\n') 
    file.write('3DOF u0 = [throttle, elev]\n')
    file.write('\n') 
    
    lysSkryf(x0, 'x0', file)
    lysSkryf(u0, 'u0', file)
    
    file.close() 


def leesGelykVlug(leernaam):
    # Lees die gelykvlug waardes in 'n *.csv leer
    #
    # Insette:
    # x0        - Beginwaardes
    # u0        - Beheerinsetwaardes by gelykvlug
    
    file = open(f'{leernaam}{'.csv'}', "r+")
    leerlyne = (file.readlines())

    for lyn in leerlyne:
        if 'x0, ' in lyn:
            lyn = lyn.replace('\n','')
            lyn = lyn.replace('x0, ','')
            lynlys = lyn.split(',')

            x0 = []
            for nommer in lynlys:
                x0.append(float(nommer))
        
        if 'u0, ' in lyn:
            lyn = lyn.replace('\n','')
            lyn = lyn.replace('u0, ','')
            lynlys = lyn.split(',')

            u0 = []
            for nommer in lynlys:
                u0.append(float(nommer))

    file.close()

    return x0, u0

def voorbeeldLeesSkryfGelykvlug():
    # x0 = [VT, ALPHA, THETA, Q, H, Distance]
    x0 = [500, 5.43*3.14159/180, 5.43*3.14159/180, 0, 30000, 0]
    # u0 = [throttle, elev]
    u0 = [0.204, -4.1]
    print('Skryf gelykvlug waardes')
    stoorGelykVlug(x0, u0, 'gelykvlugvoorbeeld', 'Toets gelykvlug string')

    x0, u0 = leesGelykVlug('gelykvlugvoorbeeld')
    print('Lees gelykvlug waardes')
    print('x0 ')
    print(x0)
    print('u0 ')
    print(u0)

#%% Druk matrikse sodat hulle maklik leesbaar is in wetenskaplike notasie


def drukMatriks(matriks):
    # Druk die lineariseerde matriksse sodat hulle maklik leesbaar is
    s = [[str("{:.4e}".format(e)) for e in row] for row in matriks]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def voorbeeldDrukMatriks():
    A = [[3.14159, 8.9e-5], [5.98, 7.2893847293742987]]
    print('Hoe matriks gewoonlik lyk')
    print(A)

    print('Hoe dit lyk in wetenskaplike notasie')
    drukMatriks(A)

#%% Main funksie

def main():
    # Voorbeeld van Runge Kutta integrasie
    print("Punt massa integrasie voorbeeld")
    voorbeeldIntegrasiePuntMassa()
    print("Pendulum integrasie voorbeeld")
    voorbeeldIntegrasiePendulum()

    # Voorbeeld van Jakobiaan berekening
    print("Jakobiaan berekening voorbeeld")
    voorbeeldJakobiaan()

    # Voorbeeld van gedempte natuurlike frekwensie en dempingberekening
    print("Natuurlike frekwensie en demping berekening voorbeeld")
    voorbeeldOmegaZeta()

    # Voorbeeld van lees en skryf van gelykvlug waardes
    voorbeeldLeesSkryfGelykvlug()

    # Voorbeeld van hoe om matriks leesbaar te druk
    voorbeeldDrukMatriks()




if __name__ == "__main__":
    main()