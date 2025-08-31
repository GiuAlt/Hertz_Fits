import jpkfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate
import math
# import seaborn as sb
 
         
import glob
 
from numpy.polynomial import polynomial as poly
from scipy import interpolate as interp


import os

cwd = os.getcwd()

def Herz(d, E, F_0):
    v = 0.5 #poisson Ratio
    R = 2.5*1e-6 #indenter radius (m)
    return  4/3 *(E * math.sqrt(R)*d**(3/2))/(1-v**2)+F_0 



directory = os.chdir("/Users/giuliaam/Desktop/Experiments/DataProject1/ViscoModel/Dataset01") # change directlry
#print("Current working directory: {0}".format(cwd))
#pd.read_excel("SP2nN.xlsx")

# jpk.get_info('segments')

k = 0.05




txtfiles = []
for file in glob.glob("*.jpk-force"):
    txtfiles.append(file)

lista = pd.DataFrame(txtfiles)    
lista.columns = ['Full']

last = lista
# last=lista['Full'].str.split('.',expand=True) 
# # ##numeratrion
# last=lista['Full'].str.split('-',expand=True)
# last['Full']= last[0]+"-" + last[1]+"-" + last[2]#+"-"+"force"
# last["Number"], last['Rest']= last[1].str.split('.',1).str
# # last = last.sort_values(by=[0,1])

# Condition = last[last[0]== "1"]
# Cell_number =1
# Location = "N"

# Condition["Bin"] = np.arange(len(Condition)) // 1
# Relevant = Condition
# # Relevant = Condition[Condition["Bin"]!=3]
# # Relevant = Relevant[Relevant["Bin"]!=0]
# # Relevant = Relevant[Relevant["Bin"]!=2]
# # 
# # Relevant = Relevant[ Relevant["Bin"] % 2 == 1]

# Relevant["Index"] = np.arange(len(Relevant))


# segments = [[0,1],[2,3],[4,5]]

# All = pd.DataFrame()


    
# for i in range (2):
    
#     print(i)
#     d = Relevant[Relevant["Index"] == i]
    
#     f = d["Full"].tolist()[0]
#     print(f) 
#     jpk = jpkfile.JPKFile(f)
 
#     Deflection = pd.DataFrame()
#     Height= pd.DataFrame()
#     TS = pd.DataFrame()
    
#     for segment in  range (0,1):
    
    
#         b = jpk.segments[0] 
#         #open first segment           
#         #b = jpk.segments[segments[segment][0]]            
#         #b = baseline.segments[segments[segment][1]]
#         app_data, app_units = b.get_array(['height', 'vDeflection']) #approach 
        
#         VDeflection_app = app_data['vDeflection']*1e9
#         height_app = app_data['height']
        

#         VDeflection_app = pd.DataFrame(app_data['vDeflection'])
        
#         height_app = pd.DataFrame(app_data['height'])
          
        
#         Deflection = pd.concat([Deflection, VDeflection_app], axis =1)
#         Deflection["Average Deflection"] = Deflection.mean(axis = 1)
        
#         Height = pd.concat([Height, height_app], axis =1)
#         Height["Average Height"] = Height.mean(axis = 1) 

  
#         Dt =[Deflection["Average Deflection"], Height["Average Height"]] 
   
#         Dt= pd.concat(Dt, axis=1)
#         Dt["Tip_sample"] = ( Dt["Average Height"]-Dt['Average Deflection']/k )*1e6


#     tip_sample = Dt["Tip_sample"].values
#     Deflection = Dt["Average Deflection"].values  
#     Deflection = Deflection[200:]
#     tip_sample = tip_sample[200:]
    

#     lD = len(Deflection)
#     end = int(lD/2)
#     x = tip_sample[0:end]
#     y = Deflection[0:end]
                     
#     z = np.polyfit(x, y, 1) 
#     Deflection = Deflection- (tip_sample*z[0] +z[1])

#     ##Deflection in nN:
#     Deflection = Deflection*1e9  

    
#     import seaborn as sb
#     f,ax = plt.subplots(1,1)
#     ax.plot(tip_sample, Deflection, color = "violet")
#     plt.xlabel("Distance (µm)")
#     plt.ylabel("Force (nN)")
#     sb.despine(ax=ax, offset=0)
#     # plt.set(ylim = (0,0.5))
#     #plt.title("Baseline Correction")
#     #plt.grid(True)   

#     B = pd.DataFrame()
#     B["Deflection"] = Deflection
#     B["TS"] = tip_sample 
#     B["Curve"] = i
   


#     All = pd.concat([All,B], axis = 0)






