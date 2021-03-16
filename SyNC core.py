# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:38:09 2020

@author: PRALAY KR CHATTERJEE
"""
import numpy as np
import math
import csv 
from scipy.integrate import odeint
from scipy.integrate import simps
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band',analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

f100 = open("temp100.txt", "r")

Isyn100s = []
RM100s = []
w100s = []
A100s = []
D100s = []
Z100s = []
RMtr100s = []
Prel100s = []
Pinh100s = []
Inh100s = []
Y100s = []
t100s = []

for line in f100:
    l = [elt.strip() for elt in line.split(' ')]
    #ks.append(float(l[1]))
    #Isyn100s.append(float(l[1]))
    #RM.append(float(l[4]))
    #C.append(float(l[5]))
    #w100s.append(float(l[2]))
    #A100s.append(float(l[3]))
    #D100s.append(float(l[4]))
    #RMtr1000s.append(float(l[5]))
    #Pinhs.append(float(l[1]))
    Prel100s.append(float(l[2]))
    Inh100s.append(float(l[1]))
    #Sm.append(float(l[13]))
    Y100s.append(float(l[0]))
    Z100s.append(float(l[0]))
    t100s.append((float(l[8])/10000000)+4)
#End of extraction of data from hardware file
print(t100s)

zx =[0,0,0,0,0,0]
tt = [0]
def I(t,tt):
    x = 0
    for tx in tt:
        x += ((t-tx)/0.1)*np.exp((-t+tx)/0.1)
    return x
def model(z,t,u,Isyn1):
    #values
    RM = 0.691
    #variables 
    G1 = 0.000000001
    #G2 = 3.2369736991
    G2 = 3.2369736991
    dsm = G1*G2
    global b
    if u == 0:
        b = 1
    if u == 1 and b==1:
        b-=1
        t0 = t
        tt.append(t0)
    #equations
    RMtrace = z[0]
    A = z[1]
    D = z[2]
    w = z[4]
    Inhx = z[3]
    Sm = z[5]
    dRMtracedt = -RMtrace/0.4 + RM
    dAdt = (-A + 1.0*u)/0.1
    dDdt = (-D + 1.0*u)/0.02
    dInhxdt = (-Inhx)/0.1 + RMtrace*A
    Pinh = np.exp((np.exp(-0.00000001/Inhx)-1)/Inhx)
    Pinh2 = -0.00007/(Inhx-0.00001)+1.1
    if Pinh2<0:
        Pinh2 = 0
    if Pinh2>1:
        Pinh2 = 1
    if Inhx < 1e-05:
        Pinh2 = 0
    Prel = (1-Pinh)*0.25
    dwdt = Prel*A*D
    Isyn = w*Isyn1
    dSmdt = (((1+np.tanh(850*(Isyn-0.001)))*(1-Sm))-Sm/dsm)/0.1
    zx[0] = Isyn
    zx[1] = Isyn
    P0[0] = Pinh
    P0[1] = Prel
    dzdt = [dRMtracedt,dAdt,dDdt,dInhxdt,dwdt,dSmdt]
    return dzdt
def model2(z2,t,Sm):
    #values
    beta = 10000000
    W = 20
    Wrest = 0
    alpha = 0.000003
    r = 0.000001
    c4 = 20
    c1 = 0.13
    c2 = 0.9
    c2p4 = 0.6561
    c3 = 0.004
    c4 = 5 
    #variables 
    #equations
    C = z2[0]
    E = z2[1]
    Ec1 = (c1*C*C)/(1+C*C)
    Ec2 = (C*C*C*C)/(c2p4+C*C*C*C)
    Ec3 = (E*E)/(1+E*E)
    Ec4 = c3*E
    dEdt = (Ec1-Ec2*Ec3-Ec4)/0.0004
    dCdt = (-C-c4*dEdt*0.04+r+alpha*(W-Wrest)+beta*(Sm))/100
    #dxdt = x-(x*x*x)/3-y+u+0.14
    #dydt = 0.08*(x+0.7-(0.8*y))
    dz2dt = [dCdt,dEdt]
    return dz2dt
def model3(z3,t,C):
    #values
    #variables 
    #equations
    x = z3[0]
    y = z3[1]
    dxdt = x-(x*x*x)/3-y+C*100000+0.14
    dydt = 0.08*(x+0.7-(0.8*y))
    dz3dt = [dxdt,dydt]
    return dz3dt

# initial condition
z0 = [0.000000001,0,0,0.000000001,0,0]
zx0 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
z20 = [0.0000001,0.0000000000001]
z30 = [0.0000001,0.0000000000001]
P0 =[0,0]

# number of time points
n = 40001

# time points
t = np.linspace(0,40,n)
noise = np.random.normal(0,1,n)
fs = 1000.0
lowcut = 150.0
highcut = 200.0

noise_filtered = butter_bandpass_filter(noise, lowcut, highcut, fs, order=6)
"""plt.plot(t[3600:4600], noise_filtered[3600:4600])
plt.xlabel('time (seconds)')
#plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.draw()"""

# impulse input
u = np.zeros(n)
for i in range(1,100):
    u[i*1000:i*1000+10]=1.0
RM = np.zeros(n)
#for i in range(1,n):
    #RM[i*1000:i*1000+10]=RM[(i-1)*1000]+0.1
    #RM[i]=0.691

# store solution

Sm = np.empty_like(t)
Isyn = np.empty_like(t)
Isynx = np.empty_like(t)
Influx = np.empty_like(t)
RMtrace = np.empty_like(t)
A = np.empty_like(t)
D = np.empty_like(t)
Inhx = np.empty_like(t)
w = np.empty_like(t)
Prel = np.empty_like(t)
Pinh = np.empty_like(t)
Cc = np.empty_like(t)
Ce = np.empty_like(t)
RM = np.empty_like(t)
V = np.empty_like(t)
W = np.empty_like(t)
# record initial conditions
RMtrace[0] = z0[0]
A[0] = z0[1]
D[0] = z0[2]
Inhx[0] = z0[3]
w[0] = z0[4]
Sm[0] = z0[5]
Pinh[0] = P0[0]
Prel[0] = P0[1] 
Isyn[0] = zx0[0] 
Isynx[0] = zx0[0] 
RM[0] = zx0[1]
V[0] = zx0[0] 
W[0] = zx0[1]
Influx[0] = 0.0
# record initial conditions

# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    Isyn1 = I(t[i-1],tt)
    # solve for next step
    z = odeint(model,z0,tspan,args=(u[i],Isyn1,))
    # store solution for plotting
    RMtrace[i] = z[1][0]
    A[i] = z[1][1]
    D[i] = z[1][2]
    Inhx[i] = z[1][3]
    w[i] = z[1][4]
    Sm[i] = z[1][5]
    Pinh[i] = P0[0]
    Prel[i] = P0[1]
    Isyn[i] = zx[0]
    z2 = odeint(model2,z20,tspan,args=(Sm[i],))
    Cc[i] = z2[1][0]
    Ce[i] = z2[1][1]
    e = 2.71828
    gf = Cc[i]*e*math.sqrt(3)*400000
    RM[i] = math.exp(-math.exp((+1.957-gf)))
    z3 = odeint(model3,z30,tspan,args=(RM[i],))
    V[i] = z3[1][0]
    W[i] = z3[1][1]
    #print (Isyn[i],t[i])
    # next initial condition
    z0 = z[1]


#plt.plot(t,Inhx,'b-',label='Inh')
plt.figure(figsize=(8.3,5.8))
#plt.plot(t[3600:4600],noise[3600:4600],'r-',label='Noise')
#ylmt = [1.02,1.02]
#xrng = [3.6,4.6]
#plt.plot(xrng,ylmt,'b--',label='Limit')
plt.plot(t[1600:4600],V[1600:4600],'m-',label='Distortion')
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('Distortion Ratio', fontsize=24)
plt.xlabel('Time (in s)', fontsize=24)
plt.legend(loc=1, fontsize=20)
plt.draw()  # Draws, but does not block

plt.figure(figsize=(8.3,5.8))
#plt.plot(t[3600:4600],noise[3600:4600],'r-',label='Noise')
plt.plot(t[1600:4600],W[1600:4600],'r-',label='Distortion')
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel('Noise', fontsize=24)
plt.xlabel('Time (in s)', fontsize=24)
plt.legend(loc=1, fontsize=20)
plt.draw()  # Draws, but does not block
plt.show()