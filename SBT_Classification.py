# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:12:21 2020
Python version: 3.7.0
Dependencies: Matplotlib, Numpy, Pandas
@author: Kieran Blacker

Description: 
    
This function should assign SBT fields using Robertson's (2010)
classification: 
    
"Robertson, P.K., 2010, May. Soil behaviour type from the CPT: 
an update. In 2nd International Symposium on Cone Penetration 
Testing (Vol. 2, pp. 575-583)."

This isn't very pythonic - I am sure it could be written much more elogently.

Past implementation made use of ray-tracing to determine point-polygon
positions (python 2.x). I decided to rewrite this function using matplotlib
"contains_points" as this seems to be easier to modify, and faster. 

Data dependency: SBT fields were digitised from Robertson (2010) using 
webplotdigitiser. The definition of these fields is somewhat arbitrary and 
I personally prefer the use of Ic, which is calculated using a circle 
projected into CPTu space. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Define classification field vertices: 

f1 = np.column_stack(([1,9.79,9.82,9.72,9.39,8.95,8.32,7.52,6.74,5.35,4.53,3.55,2.92,2.37,1.85,1.4,1.13,1,1],[1,9.79,9.82,9.72,9.39,8.95,8.32,7.52,6.74,5.35,4.53,3.55,2.92,2.37,1.85,1.4,1.13,1,1]))

f2 = np.column_stack(([1.67,1.63,1.87,2.16,2.55,3.12,3.83,4.65,5.61,6.94,8.43,9.32,10,10,1.67], [1,1.09,1.16,1.24,1.39,1.59,1.9,2.31,2.87,3.88,5.32,6.62,7.76,1,1]))

f3 = np.column_stack(([0.72,0.75,0.89,1.05,1.22,1.37,1.63,1.9,2.17,2.44,2.73,2.98,3.35,3.74,4.11,4.5,4.84,5.2,5.55,5.8,5.9,6.25,7.22,8.15,9.11,10,10,10,9.32,8.43,8.11,7.28,6.94,5.61,4.65,3.83,3.12,2.55,2.16,1.87,1.63,1.61,1.49,1.39,1.33,1.17,1.02,0.89,0.82,0.77,0.72],[4.53,4.66,5.16,5.75,6.39,7.09,8.17,9.4,10.7,12.26,14.04,15.75,18.57,21.96,25.91,30.89,36.04,42.92,50.82,58.65,63.93,62.27,58.81,57.08,56.45,56.11,25.27,7.76,6.62,5.32,4.99,4.2,3.88,2.87,2.31,1.9,1.59,1.39,1.24,1.16,1.09,1.13,1.4,1.66,1.85,2.37,2.92,3.55,3.9,4.2,4.53]))

f4 = np.column_stack(([0.31,0.38,0.53,0.71,0.92,1.13,1.4,1.7,2.03,2.4,2.73,3,3.28,3.52,3.68,3.8,3.95,4.21,4.62,5.07,5.5,5.9,5.8,5.55,5.2,4.86,4.84,4.5,4.11,3.98,3.74,3.35,2.98,2.73,2.53,2.44,2.17,2.13,1.9,1.63,1.37,1.22,1.05,0.89,0.75,0.72,0.6,0.59,0.43,0.35,0.31],[8.01,8.91,10.5,12.5,14.94,17.59,21.15,25.55,31.32,38.68,46.73,55.37,65.57,76.93,87.23,97.29,91.14,83.34,75.56,70.12,66.17,63.93,58.65,50.82,42.92,36.56,36.04,30.89,25.91,24.5,21.96,18.57,15.75,14.04,12.81,12.26,10.7,10.52,9.4,8.17,7.09,6.39,5.75,5.16,4.66,4.53,5.35,5.39,6.74,7.52,8.01]))

f5 = np.column_stack(([0.1,0.1,0.12,0.16,0.21,0.28,0.36,0.44,0.53,0.68,0.85,1,1.18,1.42,1.64,1.9,2.13,2.34,2.45,2.54,2.75,2.99,3.23,3.5,3.8,3.68,3.52,3.28,3,2.73,2.4,2.03,1.7,1.4,1.13,0.92,0.71,0.58,0.53,0.38,0.31,0.28,0.23,0.19,0.15,0.12,0.1,0.1],[26.01,25.87,26.24,27.21,29.2,31.48,34.18,37.67,41.8,48.47,56.76,65.33,75.73,91.51,110.02,137.19,168.62,203.74,232.74,214.34,181.76,152.29,130.4,111.66,97.29,87.23,76.93,65.57,55.37,46.73,38.68,31.32,25.55,21.15,17.59,14.94,12.5,11.09,10.5,8.91,8.01,8.32,8.95,9.39,9.72,9.82,9.79,26.01]))

f6 = np.column_stack(([0.1,0.1,0.12,0.14,0.19,0.25,0.3,0.38,0.45,0.53,0.61,0.68,0.78,0.86,0.94,1.02,1.36,1.47,1.62,1.89,2.17,2.45,2.34,2.13,1.9,1.65,1.64,1.42,1.18,1,0.85,0.68,0.6,0.53,0.44,0.36,0.28,0.21,0.16,0.12,0.1,0.1,0.1],[150.42,150.42,153,161.39,176.55,195.02,218.51,255.22,293.03,341.44,393.08,456.8,554.49,649.38,776.81,1000.28,1000.28,822.08,637.53,426.88,310.24,232.74,203.74,168.62,137.19,111.19,110.02,91.51,75.73,65.33,56.76,48.47,44.78,41.8,37.67,34.18,31.48,29.2,27.21,26.24,25.87,26.01,150.42]))

f7 = np.column_stack(([0.1,0.1,1.02,0.94,0.86,0.78,0.68,0.61,0.53,0.45,0.38,0.3,0.25,0.19,0.14,0.12,0.1,0.1],[150.42,1000.28,1000.28,776.81,649.38,554.49,456.8,393.08,341.44,293.03,255.22,218.51,195.02,176.55,161.39,153,150.42,150.42]))

f8 = np.column_stack(([1.36,4.76,4.73,4.72,4.64,4.56,4.52,4.4,4.32,4.22,4.12,3.97,3.8,3.5,3.23,2.99,2.77,2.75,2.54,2.45,2.17,1.89,1.62,1.47,1.36],[1000.28,1000.28,793.03,594.94,428.86,328.91,269.71,212.96,180.42,153.21,132.45,109.42,97.29,111.66,130.4,152.29,179.83,181.76,214.34,232.74,310.24,426.88,637.53,822.08,1000.28]))

f9 = np.column_stack(([3.8,3.97,4.12,4.22,4.32,4.4,4.52,4.56,4.64,4.72,4.73,4.76,10,10,10,9.11,8.15,7.22,6.25,5.9,5.5,5.07,4.62,4.21,3.95,3.8],[97.29,109.42,132.45,153.21,180.42,212.96,269.71,328.91,428.86,594.94,793.03,1000.28,1000.28,893.91,56.11,56.45,57.08,58.81,62.27,63.93,66.17,70.12,75.56,83.34,91.14,97.29]))

# Polygon field definition:
#~~~ Colour coded fields ~~~#

f1c = Polygon(f1, ec="k",fc="r", closed=True)
f2c = Polygon(f2, ec="k",fc="sienna", closed=True)
f3c = Polygon(f3, ec="k",fc="steelblue", closed=True)
f4c = Polygon(f4, ec="k",fc="teal", closed=True)
f5c = Polygon(f5, ec="k",fc="darkseagreen", closed=True)
f6c = Polygon(f6, ec="k",fc="burlywood", closed=True)
f7c = Polygon(f7, ec="k",fc="gold", closed=True)
f8c = Polygon(f8, ec="k",fc="grey", closed=True)
f9c = Polygon(f9, ec="k",fc="lightgray", closed=True)

#~~~ Polylines only ~~~#

f1w = Polygon(f1, ec="k",fc="None", closed=True)
f2w = Polygon(f2, ec="k",fc="None", closed=True)
f3w = Polygon(f3, ec="k",fc="None", closed=True)
f4w = Polygon(f4, ec="k",fc="None", closed=True)
f5w = Polygon(f5, ec="k",fc="None", closed=True)
f6w = Polygon(f6, ec="k",fc="None", closed=True)
f7w = Polygon(f7, ec="k",fc="None", closed=True)
f8w = Polygon(f8, ec="k",fc="None", closed=True)
f9w = Polygon(f9, ec="k",fc="None", closed=True)

# Classification function

def RobertsonSBT(qtn,rf):
    '''
    Inputs:
    
    qtn = Normalised corrected tip resistance (dimensionless, kPa/kPa)
    rf = friction ratio, expressed as percentage (%)
    
    Inputs can be lists or arrays, but will be converted to numpy arrays
    
    Returns:
    
    sbt = sediment behaviour type (discrete classification) as numpy array
    '''
    data = np.column_stack((qtn,rf))
    # Locate points for each classification:
    c1 = f1c.get_path().contains_points(data)*1
    c2 = f2c.get_path().contains_points(data)*2
    c3 = f3c.get_path().contains_points(data)*3
    c4 = f4c.get_path().contains_points(data)*4
    c5 = f5c.get_path().contains_points(data)*5
    c6 = f6c.get_path().contains_points(data)*6
    c7 = f7c.get_path().contains_points(data)*7
    c8 = f8c.get_path().contains_points(data)*8
    c9 = f9c.get_path().contains_points(data)*9
    # sum and stack into single array
    sbt = np.sum(np.column_stack((c1,c2,c3,c4,c5,c6,c7,c8,c9)), axis=1)
    return sbt

# SBT plotting function

def PlotSBT(qtn, rf, mode="c"):
    '''
    A plot wrapper for plotting the standard Robertson (2010) SBT chart. 
    
    Inputs: 
        
    qtn = Normalised corrected tip resistance (dimensionless, kPa/kPa)
    rf = friction ratio, expressed as percentage (%)
    mode = string; c=coloured, w=white(clear)
    
    qtn and rf can be lists or arrays, but will be converted to numpy arrays.
    '''
    # setup figure:
    fig,ax = plt.subplots()
    if mode == "c":
        ax.add_patch(f1c)
        ax.add_patch(f2c)
        ax.add_patch(f3c)
        ax.add_patch(f4c)
        ax.add_patch(f5c)
        ax.add_patch(f6c)
        ax.add_patch(f7c)
        ax.add_patch(f8c)
        ax.add_patch(f9c)
        ax.set_xlim(0.1, 10)
        ax.set_ylim(1, 1000)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Rf (%)")
        ax.set_ylabel("Qtn (dimensionless)")
        ax.plot(qtn,rf, "ok")
        plt.show()
    elif mode == "w":
        ax.add_patch(f1w)
        ax.add_patch(f2w)
        ax.add_patch(f3w)
        ax.add_patch(f4w)
        ax.add_patch(f5w)
        ax.add_patch(f6w)
        ax.add_patch(f7w)
        ax.add_patch(f8w)
        ax.add_patch(f9w)
        ax.set_xlim(0.1, 10)
        ax.set_ylim(1, 1000)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Rf (%)")
        ax.set_ylabel("Qtn (dimensionless)")
        ax.plot(qtn,rf, "ok")
        plt.show()
    else:
        print("ERROR - Please specify plot mode")
    
# test data: 
        
qtn, rf = np.array([1,0.5,0.7,8]), np.array([5,30, 90, 18])
sbt = RobertsonSBT(qtn, rf)
print(sbt)
testplot = PlotSBT(qtn, rf, mode="c")
    
    