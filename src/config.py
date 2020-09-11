import math
import numpy as np

#Simulation Parameters
TotalSteps = 40000 #10000
TimeStep = 0.00001 #seconds --> maybe make it smaller with more steps; large errors accrue #0.001

###try smaller distances/masses so that the swing is not sooooo big

#System parameters
PI = math.pi #Converts the object math.pi --> PI (a more useable name)
g = 9.81 #m/s^s #Gravitational acceleration
DragCoefficient = 1
DragCoef = 1.0001 #Use this drag coefficient implementation (not the one above)
#QUESTION So you must do this for each particle youre iitializing?
Position = [np.array([1,0,0]),np.array([3,0,0]),np.array([2,1,2])]   #Define an array of positions of N spherical particles (with respect to origin) rigidly connected; each element is an ordered triple
Mass = [.2,0.1,.2]       #Defines the masses of each of the N spherical particles
TotalMass = sum(Mass) #Sums the masses of the N particles
print("Total Mass: \n", TotalMass)
Radius = [1,2,2]
CartesianBasis = np.identity(3)
N = len(Mass)
