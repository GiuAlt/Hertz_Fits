
import jpkfile
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate
import math
import seaborn as sb
import statistics
from numpy.polynomial import polynomial as poly
from scipy import interpolate as interp
from scipy import stats
import glob
##Directory
import os
cwd = os.getcwd()

def func0(x,a,b,beta): ### power law
     return a + b*x**beta  
def func1(x,a,tau,c):
     return a * np.exp(-x/tau)+ c  
    
def poroEl(x,ap0,ap1,Dp,L):
     return ap0+ap1*np.exp(-Dp*x/L**2)

         ##Double exponential fit
def func2(x,a0,a1,tau1,a2,tau2):  
      return a0 +(a1 * np.exp(-x/tau1))+(a2 * np.exp(-x/tau2))
  
def func3(x,a0,a1,tau1,a2,tau2,a3,tau3):  
      return a0 + (a1 * np.exp(-x/tau1)+a2 * np.exp(-x/tau2)+a3 * np.exp(-x/tau3))


def Herz(d, E, F_0):
    v = 0.5 #poisson Ratio
    
    R = 2.5*1e-6 #indenter radius (m)
    return  4/3 *(E * math.sqrt(R)*d**(3/2))/(1-v**2)+F_0 

Treatment = ""
arr1 = []
# k = 0.073
raggioB = 2
arr = []
width = 50/2.54
height = 25/2.54

#### pick up the cell you need
Conf = Condition[Condition["Bin"]==0]
f = Conf["Full"].tolist()[0]
jpk = jpkfile.JPKFile(f)
 
### choose the segment  
r = 3
approach = jpk.segments[1]              
# retract = jpk.segments[1]

## select the signals you need
app, app_units = approach.get_array(['vDeflection']) #approach
VDeflection = app['vDeflection']*1e9
conf_time = approach.get_array(['t'])
conf_time = conf_time[0]
time1 = np.asarray(conf_time).astype(float).flatten() 

### Linear fit of the part where the cantilever is not in contact

confinement= jpk.segments[r-1] 
lD = len(VDeflection)
end = int(lD/1) 
mi = 0
x = time1[mi:end].flatten()
y = VDeflection[mi:end].flatten()
z = np.polyfit(x, y, 1) 

#Baseline correction of the indentation curve with the fit from the non contact part   
Deflection1 = VDeflection#- (time1*z[0] +z[1]) 
Deflection1 = Deflection1.flatten()


confinement= jpk.segments[r-1]    
  #approach = jpk.segment[segments[segment][0]]   

conf_data, conf_units = confinement.get_array(['vDeflection'])  #approach

Deflection = conf_data['vDeflection']*1e9
Deflection2 =Deflection.flatten()


  #DeflApp = app_data['vDeflection']*1e9
conf_time = confinement.get_array(['t'])
conf_time = conf_time[0]
time2 = np.asarray(conf_time).astype(float).flatten()  

## Correction of data with linear fit from 
Deflection2 = (Deflection2 -(+z[1]))

Time3 = time2#[time2<295] ### the whole 10 s fit.
Deflection3 = Deflection2[0:len(Time3)]

## peack force 
Top = max(Deflection2)#.flatten()
# Time3 = Time3*1e3
x = Time3#[0:lD:100]#[0:60000:100]#[:30000:100]
y = Deflection3#.flatten()#[:30000].flatten()


sigma = np.ones(len(x))
sigma[[0, -1]] = 0.01

## power law
fit0, pvoc0 =curve_fit(func0, x, y,sigma = sigma, maxfev = 10000000)#,
#fit1, pcov1 = curve_fit(func1, x, y, sigma = sigma, maxfev = 10000000)#,
fit2, pcov2 = curve_fit(func2, x, y,sigma = sigma, maxfev = 10000000)#,
#fit3, pcov3 = curve_fit(func3, x, y,sigma = sigma, maxfev = 10000000)
fit4, pcov4 = curve_fit(poroEl, x, y,sigma = sigma, maxfev = 10000000)#,

DeflectionL = np.log([Deflection3])
#DeflectionL = np.flip(DeflectionL)
DeflectionL = DeflectionL.flatten()
#make linear fit over the curve

timeL = np.log(Time3) 

 # 
f,ax = plt.subplots(1,1)
ax.plot(Time3, Deflection3)
ax.plot(x, func0(x, *fit0),label = "single power-law")
ax.plot(x, func2(x, *fit2),label = "double power-law")
ax.plot(x, poroEl(x, *fit4),label = "Poroelastic fit")
plt.legend()

### Single decay fit #################
width = 8/ 2.54
height = 16/ 2.54

fig, ax2 = plt.subplots(1,1,dpi= 300)

  #ax.plot(time, DeflApp)
ax2.plot(Time3, Deflection3)
ax2.plot(x, func0(x, *fit0), label = "single power-law")
ax2.plot(x, func2(x, *fit2), label = "double power-law")
ax2.plot(x, poroEl(x, *fit4), label = "Poroelastic fit")

ax2.set_yscale('log')
ax2.set_xscale('log')
plt.legend()
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Deflection (nN)")
# plt.savefig('/Users/giuliaam/Desktop/Experiments/DataProject1/Panel5/F4/∆H/Poroelastic/PoroelasticFit_20.pdf',dpi=300, bbox_inches = "tight")
# 
EndForce = Deflection2[-1]
print(EndForce)
# arr1.append([popt3[0],popt3[1], popt3[2], popt3[3],popt3[4], popt3[5],popt3[6], EndForce,Top])
arr1.append([Cell_number,fit0[2],fit2,fit4, EndForce,Top, Treatment])


data2 = pd.DataFrame(arr1)
# data2.columns =["a","a1","tau1", "a2","tau2", "a3", "tau3", "Max","EndForce"]
data2.columns =["Cell number","beta","fit2","Poroel", "Endforce","Peakforce", "Treatment"]
totalD = pd.DataFrame()
totalD = pd.concat([totalD, data2], axis =0)   

# totalD.to_excel('/Users/giuliaam/Desktop/Experiments/Richi01.xlsx',sheet_name='Sheet1')    
# np.save("C1.npy",Deflection2)
    