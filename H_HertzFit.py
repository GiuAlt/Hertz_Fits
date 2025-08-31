#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 12:13:38 2022

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
 
from numpy.polynomial import polynomial as poly
from scipy import interpolate as interp

#import jpkfile
#### Get current imput directory######
# Import the os modul0
import os
#skript3.groupby(['Waiting time', 'Moment'])['Norm'].mean()

cwd = os.getcwd()

def Herz(d, E, F_0):
    v = 0.5 #poisson Ratio
    R = 2.5*1e-6 #indenter radius (m)
    return  4/3 *(E * math.sqrt(R)*d**(3/2))/(1-v**2)+F_0 

# dat1 = pd.read_excel("/Users/giuliaam/Desktop/Experiments/220603/DMSOData/P2M1.xlsx")
############CHANGE THE FITTING LENGTH ACORDING TO THE CELL LINE!!!!!!
##### !1!!######4###D

##########5##97# Pick the curve: 
t = 0
d = Condition[Condition["Bin"] ==3]
# d = Relevant[Relevant["Index"] ==0]
d = d["Full"].tolist()[0]
#

Glass = jpkfile.JPKFile(d)
# # Glass = jpkfile.JPKFile("/Volumes/giuliaam/2023Experiments/230118/BAPTA-AM/Glass1-0007.jpk-force")
# CPxG = "NaN"
########## General parameters

data = All[All["Curve"] ==t]

Moment = t#[0,5,20,30,60,120,180,240,300]

CompressionForceStandard = 2

# ConfinementDepth = "NaN"
CompressionForce = 16
fit0 = "NaN"
fit1 ="NaN"
fit2="NaN"
fit3 = "NaN"
fit4 = "NaN"
EndForce = "NaN"
Top = "NaN"

CT = 10
Replicate = 3
CompressionSpeed = 5

raggioB = 1
Intervals = t#[0, 0.1, 5,10,30,60,120,180,240,300] 
#Intervals = [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
# Intervals = list(Relevant["Bin"])
# Intervals = 

# Interval = 0
Interval = t#Intervals[t]

arr = []
total = pd.DataFrame()

Treatment = "DMSO"
# Location = "Nucleus"
Line = "CAF"


# ------------------------------------------------------------------------------
#Glass contact point:
# # #------------------------------------------------------------------------------
GlassSegments = [[0,1]]
approach = Glass.segments[0]              
retract = Glass.segments[1]
app_glass, app_units = approach.get_array(['height', 'vDeflection']) #approach
ret_glass, ret_units = retract.get_array(['height', 'vDeflection']) #retract   
# Glass.segments 
# print(Glass.get_info('segments'))
#  
VDeflection_glass = app_glass['vDeflection']
height_glass = app_glass['height']
tsG = (height_glass-VDeflection_glass/k)*1e6
   
#make linear fit over the curve
x = tsG[0:500].flatten()
y = VDeflection_glass[0:500].flatten()
z = np.polyfit(x, y, 1) 

#Baseline correction:    
DeflectionGlass = VDeflection_glass- (tsG*z[0] +z[1]) 
DeflectionGlass = DeflectionGlass*1e9

# peak = DeflectionGlass.std()
# peak = peak*6
    
peak = 1
CPxG = tsG[DeflectionGlass > peak][0]
CPyG = DeflectionGlass[DeflectionGlass > peak][0]

# #CPxG = 67.5

#plot the fit:
f,ax = plt.subplots(1,1)
ax.plot(tsG, DeflectionGlass, color='orange')
ax.plot(CPxG ,CPyG, marker='o', color='r' )
plt.grid(True) 



####################### cell: 
Deflection = data["Deflection"].values 
tip_sample = data["TS"].values

lD = len(Deflection)
end = int(lD/2)

peak = Deflection[0:end].std()
peak = 0.01
Deflection= Deflection[Deflection<0.51]

tip_sample = tip_sample[0:len(Deflection)]
i = True

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
        # 9
# CPx =54.25
# CPy = 0
width = 12/ 2.54
height = 12 / 2.54
fig, ax = plt.subplots(1,1,dpi= 300)
fig.set_size_inches(width, height)   
ax.plot(tip_sample, Deflection, color = "g")
plt.xlabel("tip-sample separation (µm)")
ax.plot(CPx, CPy, marker='o', color='r' )
plt.ylabel("Force (nN)")
plt.title("Baseline Correction")
plt.grid(True)      
Height_cell = "NaN"#CPx- CPxG 
plt.savefig('/Users/giuliaam/Desktop/Experiments/Figures/Fit.svg',dpi=300, bbox_inches = "tight")



# Tip_centered = (tip_sample - CPx)
# fig, ax = plt.subplots(1, 1, dpi=300)
# ax.plot(Tip_centered, Deflection, color='violet', label='Approach')
# #plt.plot(tip_sample_Ret, Force_Ret_aligned, color='orange', label='Retract')
# plt.xlabel("Distance (µm)")
# plt.ylabel("Force (nN)")
# plt.title("")
# sb.despine(ax=ax, offset=0)
# #plt.grid(True)
# plt.legend()
# #plt.savefig('/Users/giuliaam/Desktop/Experiments/Figures/Fit.svg',dpi=300, bbox_inches = "tight")




# plt.savefig('/Users/giuliaam/Desktop/Experiments/DataProject1/Panel5/F1/Supplementary1/Curves/C2.svg',dpi=300, bbox_inches = "tight")

d_app_sepFinal = tip_sample - CPx

s = Deflection[d_app_sepFinal <= 0].flatten()
D =  np.abs(d_app_sepFinal[d_app_sepFinal <= 0]).flatten()

IndentationDepth = max(D)
# MaxInd =0.6815806925040775

if IndentationDepth < 0.8:
    Maxind = IndentationDepth
else:
    Maxind = 0.8
    
# Maxind = IndentationDepth    

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


Tip_centered = (tip_sample - CPx)
fig, ax = plt.subplots(1, 1, dpi=300)
ax.plot(Tip_centered, Deflection, color='violet', label='Approach')
ax.plot(-1*dFinal,FFinal,label='Fitting region')
ax.plot(0, CPy, marker='o', color='black' )
#ax.plot(-1*x_line,y_line, label = 'Hertz fit')
#plt.plot(tip_sample_Ret, Force_Ret_aligned, color='orange', label='Retract')
plt.xlabel("Distance (µm)")
plt.ylabel("Force (nN)")
plt.title("")
sb.despine(ax=ax, offset=0)
plt.savefig('/Users/giuliaam/Desktop/Experiments/Figures/Fit.svg',dpi=300, bbox_inches = "tight")
#plt.grid(True)
plt.legend()

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
SetpointForce = max(Deflection)
Height_cell = CPx- CPxG 
Error = np.diag(pcovF)
CH0 =0
Strain = "NaN"#(Height_cell - CH0)/CH0
Norm = poptF[0]/1196.1833345250086
arr.append( [Treatment,Line,Location,CompressionForce, Moment, Cell_number ,IndentationDepth, CantileverDeflection,
             poptF[0],Norm,Interval, FittingLength,  CPx, CT,  Replicate, Height_cell, ConfinementDepth ,SetpointForce, 
             CompressionForceStandard, CompressionSpeed,fit0[2],fit2,fit4, EndForce,Top, Strain,Error, Location])
print("Final data: ", arr)
   
   
Base = pd.DataFrame()
Base = pd.DataFrame(arr)
Base.columns =["Treatment","Line", "Location","SetpointCompr", "Moment", "Cell number", "Indentation depth", "Cantilever Deflection",  
               "E", "Norm","Interval","Fitting Length", "Contact Point","Contact time", "Replicate", "Cell height", "Compression depth","Setpoint", 
               "Force of compression", "Compression Speed","beta","fit2","fit3", "Endforce","Max","Strain","Error", "Location"]
print("Data: ", Base)
   

print(Height_cell)
print(poptF[0])  

  
# Norm =  poptF[0] / 130.20213246867743
# arr.append( [Treatment,Line,Location,  Setpoint, Moment, Cell ,IndentationDepth, CantileverDeflection,poptF[0],Norm,Interval, FittingLength,  CPx, CT,  Replicate, Height_cell,ConfinementDepth, CompressionForce ])
# #print("Final data: ", arr)
   
   
# Base = pd.DataFrame()
# Base = Base.append(arr)
# Base.columns =["Treatment","Line", "Location","Setpoint", "Moment", "Cell number", "Indentation depth", "Cantilever Deflection",  "E", "Norm","Interval","Fitting Length", "Contact Point","Contact time", "Replicate","Cell Height", "ConfinementDepth", "Compression Force"]
# print("Data: ", Base)
   
total = pd.concat([total, Base], axis =0)

# print(Norm)
# print( poptF[0])

