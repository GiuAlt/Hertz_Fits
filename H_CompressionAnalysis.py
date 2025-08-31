#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:08:39 2022

@author: giuliaam
"""
import jpkfile
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from scipy import interpolate
import math
import seaborn as sb
 
         
import glob
 
from numpy.polynomial import polynomial as poly
from scipy import interpolate as interp



import os

cwd = os.getcwd()
def Herz(d, E, F_0):
    v = 0.5 #poisson Ratio
    R = 2.5*1e-6 #indenter radius (m)
    return  4/3 *(E * math.sqrt(R)*d**(3/2))/(1-v**2)+F_0 
#Height_cell = "nGlassRef"
# ------------------------------------------------------------------------------
############### Confinement analysis#########################################

# ------------------------------------------------------------------------------
raggioB = 1
arr = []
total = pd.DataFrame()
segments = [0,1]

Conf = Condition[Condition["Bin"]==0]

f = Conf["Full"].tolist()[0]
jpk = jpkfile.JPKFile(f)

jpk = jpkfile.JPKFile(f)
  
approach = jpk.segments[0]              
# retract = jpk.segments[1]


#app, app_units = approach.get_array(['measuredHeight', 'vDeflection']) #approach
app, app_units = approach.get_array(['height', 'vDeflection']) #approach

VDeflection = app['vDeflection']
height = app['height']

x = height[700:1500].flatten()
y = VDeflection[700:1500].flatten()
z = np.polyfit(x, y, 1) 

#Baseline correction:    
Deflection = VDeflection- (height*z[0] +z[1]) 

  
tip_sample = (height-Deflection/k)*1e6
Deflection = Deflection*1e9  
lD = len(Deflection)
end = int(lD/10)

 
#### Nanowizaard 5 
# t = 1
# data = All[All["Curve"] ==t]

# Deflection = data["Deflection"].values 
# tip_sample = data["TS"].values


# peak = Deflection[0:end].std()
# peak = 0.03


# # peak = 0.003
# f,ax = plt.subplots(1,1)
# ax.plot(tip_sample, Deflection, color = "g")
# plt.xlabel("tip-sample separation (µm)")
# plt.grid(True) 

# i = True
    
#while r ==True: 
while i == True:
      
        
        CPx = tip_sample[Deflection > peak][0]
        CPy = Deflection[Deflection > peak][0]
       
        #contact_point = contact_points[speed] 
        d_app_sep = tip_sample - CPx
        
        F = Deflection[d_app_sep <= 0].flatten()
        d =  np.abs(d_app_sep[d_app_sep <= 0]).flatten()
        
        Setpoint = max(F)
        popt, pcov = curve_fit(Herz, d, F)
       
        #Create the fitted line
        x_line = np.linspace(0,max(d),len(F))
        y_line = Herz(x_line, popt[0],popt[1])
        #calculate RMSE
        y_actual = F
        y_predicted = y_line
        
        MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
        
        RMSE = math.sqrt(MSE)
        print("RMSE: ", RMSE)
           
        if RMSE > raggioB:
            peak = peak +0.001
           
        else:
            
            i = False

        print("Elastic modulus: ", popt[0],popt[1])
        print(np.sqrt(np.diag(pcov)))
           
        #Create the fitted line
        x_line = np.linspace(0,max(d),len(F))
        y_line = Herz(x_line, popt[0],popt[1])
            
CPx = 55
CPy = 0
f,ax = plt.subplots(1,1)
ax.plot(tip_sample, Deflection, color = "g")
plt.xlabel("tip-sample separation (µm)")
ax.plot(CPx, CPy, marker='o', color='r' )
plt.ylabel("Force (nN)")
plt.title("Baseline Correction")
plt.grid(True)      
# Height_cell = CPx- CPxG 
# print("The cell is ", Height_cell, "µm")
   

d_app_sepFinal = tip_sample - CPx

s = Deflection[d_app_sepFinal <= 0].flatten()
D =  np.abs(d_app_sepFinal[d_app_sepFinal <= 0]).flatten()

IndentationDepth = max(D)
ConfinementDepth= max(D)
if IndentationDepth < 0.8:
    Maxind = IndentationDepth
else:
    Maxind = 0.8    

FL = CPx - Maxind

DF = Deflection[tip_sample <= FL][0]

SetPoint = s[s > DF]
maxInd = s[s < DF]
SP = len(maxInd)


#Define array for the fit:
FFinal = s[0:SP]
dFinal = D[0:SP]

FittingLength = max(dFinal)- min(dFinal)


poptF, pcovF = curve_fit(Herz, dFinal, FFinal)
   
#Create the fitted line
x_line = np.linspace(0,max(dFinal),len(FFinal))
y_line = Herz(x_line, poptF[0],poptF[1])
#calculate RMSE
y_actual = FFinal
y_predicted = y_line

MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 

RMSE = math.sqrt(MSE)
print("RMSE: ", RMSE)            
   
#plot the fit:
f,ax = plt.subplots(1,1)
ax.plot(dFinal,FFinal)
ax.plot(x_line,y_line, label = 'Fitted line')

#------------------------------------------------------------------------------
#Find indentation depth:
#------------------------------------------------------------------------------
cDefl = (Deflection/k)#nanometri   
#print (cDefl)
Distance = min(d_app_sep)
print(Distance)
Cut = max(F)
   
# #Extract parameters:
CantileverDeflection = max(cDefl)
CompressionForce = max(Deflection)


  
  
Norm =  1#poptF[0] #/417.9862476285136
arr.append( [IndentationDepth, CantileverDeflection,poptF[0],Norm, FittingLength,  CPx, ConfinementDepth])
#print("Final data: ", arr)
   
   
Base = pd.DataFrame()
Base = Base.append(arr)
Base.columns =["Indentation depth", "Cantilever Deflection",  "E", "Norm","Fitting Length", "Contact Point", "Confinement Depth"]
print("Data: ", Base)
   
total = pd.concat([total, Base], axis =0)
print(ConfinementDepth)

#ConfinementDepth = 2