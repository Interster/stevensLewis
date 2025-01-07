# Numeriese Jakobiaan berekeninge

Hier is 'n goeie verduideliking van die numeriese Jakobiaan:

https://aleksandarhaber.com/correct-and-clear-explanation-of-linearization-of-dynamical-systems/

Hier is ook 'n voorbeeld van hoe om dit in SymPy simbolies te evalueer:

https://aleksandarhaber.com/symbolic-and-automatic-linearization-of-nonlinear-systems-in-python-by-using-sympy-library/



Die numeriese Jakobiaan word gebruik vir linearisasie van die bewegingsvergelykings rondom 'n bedryfspunt wat gewoonlik 'n gestadigde toestand is.  Die eiewaardes van die $A$ matriks word ook gebruik om die natuurlike frekwensie en dempingsverhoudings van die stelsel rondom die bedryfspunt te bereken.



Die Jakobiaan van 'n toestandsveranderlike vektor word gedefinieer as:


$$
A = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \ldots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \ldots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots &  & \vdots \\
\frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \ldots & \frac{\partial f_n}{\partial x_n}
\end{bmatrix}
,
B = \frac{\partial \mathbf{f}}{\partial \mathbf{u}} = 
\begin{bmatrix}
\frac{\partial f_1}{\partial u_1} & \frac{\partial f_1}{\partial u_2} & \ldots & \frac{\partial f_1}{\partial u_m} \\
\frac{\partial f_2}{\partial u_1} & \frac{\partial f_2}{\partial u_2} & \ldots & \frac{\partial f_2}{\partial u_m} \\
\vdots & \vdots &  & \vdots \\
\frac{\partial f_n}{\partial u_1} & \frac{\partial f_n}{\partial u_2} & \ldots & \frac{\partial f_n}{\partial u_m}
\end{bmatrix}
$$


Skryf 'n python funksie sodat die volgende vektor vir elke veranderlike bereken word tot 'n toleransie van 0.001.


$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x_n}} = 
\begin{bmatrix}
\frac{\partial f_1}{\partial x_n}  \\
\frac{\partial f_2}{\partial x_n}  \\
\vdots \\
\frac{\partial f_n}{\partial x_n}  
\end{bmatrix}
$$


Gebruik die pendulum toestandsveranderlike model om die Jakobiaan funksie te ontfout/toets.  Die pendulum se bewegingsvergelykings is as volg:


$$
\begin{bmatrix}
\dot{x_1}  \\
\dot{x_2} 
\end{bmatrix}
=
\begin{bmatrix}
f_1  \\
f_2 
\end{bmatrix}
=
\begin{bmatrix}
x_2  \\
-\frac{g}{l} \sin(x_1) + \frac{1}{ml} u^2
\end{bmatrix}
$$
waar $l$ die lengte van die pendulum is en $m$ die massa is.

waar $\dot{x_1} = \dot{\theta}$ en $\dot{x_2} = \ddot{\theta}$ en $\theta$ is die hoek van die pendulum tou met betrekking tot die vertikale as.

Die krag $F$ op die pendulum word gelyk gestel aan:
$$
F = u^2
$$
om latere differensiasie te vergemaklik.  Dit is hoe die bewegingsvergelyking hierbo bepaal word.

Die simboliese evaluering van die Jakobiaan van hierdie stelsel is as volg:
$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} \\
\end{bmatrix} = 
\begin{bmatrix}
0 & 1 \\
-\frac{g}{l} \cos(x_1) & 0 \\
\end{bmatrix}
$$
Vir die ekwilibrium punt van $\begin{bmatrix} x_1 \\ x_2\end{bmatrix} = \begin{bmatrix} 0 \\ 0\end{bmatrix}$ , of waar die pedulum stil vertikaal hang, is die waarde van die Jakobiaan:
$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
0 & 1 \\
-\frac{g}{l} \cos(x_1) & 0 \\
\end{bmatrix} = 
\begin{bmatrix}
0 & 1 \\
-\frac{9.81 m/s^2}{3m} \cos(0^\circ) & 0 \\
\end{bmatrix} =
\begin{bmatrix}
0 & 1 \\
-3.27 & 0 \\
\end{bmatrix}
$$

Die Jakobiaan matriks met betrekking tot die beheervektor $\mathbf{u}$ kan as volg bepaal word:

Onthou dat daar net een beheer inset is dus is $\mathbf{u} = u_1 = u$:
$$
\frac{\partial \mathbf{f}}{\partial \mathbf{u}} = 
\begin{bmatrix}
\frac{\partial f_1}{\partial u} \\
\frac{\partial f_2}{\partial u}
\end{bmatrix}
=
\begin{bmatrix}
0 \\
\frac{2}{ml} u \\
\end{bmatrix}
=
\begin{bmatrix}
0 \\
\frac{2}{ml} \sqrt F \\
\end{bmatrix} = 
\begin{bmatrix}
0 \\
\frac{2}{(0.1kg)(3m)} \sqrt F \\
\end{bmatrix}
$$

Vir $F = 0.5N$: 
$$
\frac{\partial \mathbf{f}}{\partial \mathbf{u}} = 
\begin{bmatrix}
0 \\
\frac{2}{(0.1kg)(3m)} \sqrt{0.5} \\
\end{bmatrix} 
=
4.71
$$

