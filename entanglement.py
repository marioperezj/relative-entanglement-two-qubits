import numpy as np
import math as math
import scipy as sci
import random as ran
from numpy import *
from math import log
from numpy import linalg as LA
from numpy.linalg import inv
_epsilon = sqrt(finfo(float).eps)
ket0=np.matrix('1;0')
ket1=np.matrix('0;1')
ket00bra00=np.outer(np.kron(ket0,ket0),np.kron(ket0,ket0).H)
def logi(x):         #funcion que calcula el logaritmo de una matriz 
  y=np.zeros_like(x) #entrada por entrada. Ademas define log0=0
  for i in np.arange(0,x.shape[0]):
    for j in np.arange(0,x.shape[1]):
      if np.around(x[i,j],decimals=13)==0:
         y[i,j]=0
      else:
         y[i,j]=np.log(np.around(x[i,j],decimals=13))
  return y
sigma=np.empty([4,4])
print "Este es un codigo que calcula el enredamiento de un sistema;\
       de dos quibits"
print "A continuacion ingresa los diez elementos necesarios para ;\
       definir una matriz de densidad expresada en la base computacional "
sigma[0,0]=input("Ingresa la entrada 0,0 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[0,1]=input("Ingresa la entrada 0,1 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[0,2]=input("Ingresa la entrada 0,2 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[0,3]=input("Ingresa la entrada 0,3 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[1,0]=conj(sigma[0,1])
sigma[1,1]=input("Ingresa la entrada 1,1 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[1,2]=input("Ingresa la entrada 1,2 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[1,3]=input("Ingresa la entrada 1,3 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[2,0]=conj(sigma[0,2])
sigma[2,1]=conj(sigma[1,2])
sigma[2,2]=input("Ingresa la entrada 2,2 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[2,3]=input("Ingresa la entrada 2,3 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
sigma[3,0]=conj(sigma[0,3])
sigma[3,1]=conj(sigma[1,3])
sigma[3,2]=conj(sigma[2,3])
sigma[3,3]=input("Ingresa la entrada 3,3 de la matriz de densidad a la cual;\
                 se le calculara el enredamiento:")
print ("A continuacion se muestra la matriz ingresada:")
s = [[str(e) for e in row] for row in sigma]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print ('\n'.join(table))
def chabas(x,y):       #funcion que aplica un cambio de base de la matriz x 
    z=np.zeros_like(x) #con el cambio y
    z=np.dot(inv(y),np.dot(x,y))
    return z
def logm(x): #funcion  que calcula el logaritmo de una matriz invertible
    w,v=LA.eig(x)
    z=np.zeros_like(x)
    z=chabas(logi(chabas(x,v)),inv(v))
    return z
def S(x): 
    return (np.trace(np.dot(sigma,logm(sigma)))-np.trace(np.dot(sigma,logm(den(x))))).real
def S1(x): 
    return (np.trace(np.dot(sigma,logm(sigma)))-np.trace(np.dot(sigma,logm(x)))).real
def prod(x): #funcion que ejecuta el producto de todos los elementos de una lista
    prod=1
    for i in np.arange(0,x.size):
        prod=prod*x[i]
    return prod
def ran(x): #funcion que genera una lista de longitud x de numeros aleatorios 
    a=[random.uniform(0,2*pi) for _ in range(0,x)]
    b=np.array(a)
    return b
def ran1(x): #funcion que genera una lista de longitud x de numeros aleatorios
    a=[random.uniform(0,0.001) for _ in range(0,x)]
    b=np.array(a)
    return b
phi0=np.array([pi/2])
phi=np.append(phi0,ran(15))
alpha=ran(16)
beta=ran(16)
eta=ran(16)
mu=ran(16)
vari=0
a=np.append(beta,eta)
b=np.append(phi,alpha)
c=np.append(b,a)
rho=np.append(c,mu)
def P(x,y): #funcion que evalua los p_i cuadrados
    prod=1
    for i in np.arange(y,16):
        prod=prod*cos(x[i])
    p=sin(x[y-1])*prod
    return p
def psi1(a,k):
    return cos(a[k+15])*ket0+sin(a[k+15])*exp(complex(0,a[k+47]))*ket1
def psi2(a,k):
    return cos(a[k+31])*ket0+sin(a[k+31])*exp(complex(0,a[k+63]))*ket1
def den(y):
    sum=np.zeros_like(sigma)
    for i in np.arange(1,17):
        sum=sum+np.square(P(y,i))*np.kron(np.outer(psi1(y,i),psi1(y,i).H),np.outer(psi2(y,i),psi2(y,i).H))
    return sum
def consum(y):
    for i in np.arange(1,17):
        print np.square(P(y,i))*np.kron(np.outer(psi1(y,i),psi1(y,i).H),np.outer(psi2(y,i),psi2(y,i).H))
def Grad(x):
    vari=ran1(1)
    gra=np.zeros_like(rho)
    for i in np.arange(1,80):
        vari2=np.zeros_like(x)
        vari2[i]=vari
        rhop=x+vari2
        rhom=x-vari2
        gra[i]=(S(rhop)-S(rhom))*(1/(2*vari))
    return gra
############################################################
def BacktrackingLineSearch(f, df, x, p, df_x = None, f_x = None, args = (),alpha = 0.01, beta = 0.8, eps = _epsilon, Verbose = False):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df_x: gradient at x
    f_x = f(x) (Optional)
    args: optional arguments to f (optional)
    alpha, beta: backtracking parameters
    eps: (Optional) quit if norm of step produced is less than this
    Verbose: (Optional) Print lots of info about progress
    
    Reference: Nocedal and Wright 2/e (2006), p. 37
    
    Usage notes:
    -----------
    Recommended for Newton methods; less appropriate for quasi-Newton or conjugate gradients
    """

    if f_x is None:
        f_x = f(x, *args)
    if df_x is None:
        df_x = df(x, *args)

    assert df_x.T.shape == p.shape
    assert 0 < alpha < 1, 'Invalid value of alpha in backtracking linesearch'
    assert 0 < beta < 1, 'Invalid value of beta in backtracking linesearch'

    derphi = dot(df_x, p)

    assert derphi.shape == (1, 1) or derphi.shape == ()
    assert derphi < 0, 'Attempted to linesearch uphill'

    stp = 1.0
    fc = 0
    len_p = LA.norm(p)


    #Loop until Armijo condition is satisfied
    while f(x + stp * p, *args) > f_x + alpha * stp * derphi:
        stp *= beta
        fc += 1
        if Verbose: print('linesearch iteration'), fc, ':', stp, f(x + stp * p, *args), f_x + alpha * stp * derphi
        if stp * len_p < eps:
            print('Step is  too small, stop')
            break
    #if Verbose: print'linesearch iteration 0 :', stp, f_x, f_x

    if Verbose: print('linesearch done')
    #print fc, 'iterations in linesearch'
    return stp
#################################################################
rhold=np.zeros_like(rho)
iter=input("Ingresa numero de iteraciones maximo ")
sum=1
itera=0
while (sum > 0.0001 and itera < iter):
    rhold=np.copy(rho)
    gradi=Grad(rho)
    best=BacktrackingLineSearch(S,Grad,rho,-gradi)
    rho=rhold-best*gradi
    sum=0
    for k in np.arange(0,3):
        for l in np.arange(0,3):
            sum=sum+absolute(den(rhold)[k,l]-den(rho)[k,l])
    itera=itera+1
    print itera
rhoround=np.around(den(rho),decimals=4)
print ("La matriz desenredada mas cercana es: ")
s1 = [[str(e) for e in row] for row in rhoround]
lens1 = [max(map(len, col)) for col in zip(*s1)]
fmt1 = '\t'.join('{{:{}}}'.format(x) for x in lens1)
table1 = [fmt1.format(*row) for row in s1]
print '\n'.join(table1)
print ("El enredmaiento de la matriz es: ")
print (S(rho))
print ("La ultima iteracion tuvo una diferencia menor en ;\
       cada una de sus entradas a ")
print sum
