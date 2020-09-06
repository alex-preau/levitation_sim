import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as rot
import rowan as quat
import math
from src.config import N,Mass,CartesianBasis,TimeStep,TotalMass
from src.levitation_forces import Drag
#Moment of Interia Matrix with respect to the COM

#Need to fill diagonal elements differently to off-diagonal elements; the matrix is also symmetric [I]_ij = [I]_ji;
#It's only a 3x3 matrix, so these while loops shouldn't be too expensive. Maybe there's a better way in python?
def InertiaTensorFill(Positions):
    #Globally initialize an empty 3x3 matrix
    Inertia = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i != j and Inertia[i][j] == 0 and Inertia[j][i] == 0: #For the off-diagonal elements, it will fill them if non-zero; we have initialized a 0 matrix in M3x3
                k = 0
                print(Inertia[i][j])
                while k < N: #Sums over all masses and their projections onto a combination of two axes
                    Inertia[i][j] += Mass[k] * np.dot(Positions[k], CartesianBasis[i]) * np.dot(Positions[k], CartesianBasis[j])
                    Inertia[j][i] = Inertia[i][j] #Because it is a symmetric tensor
                    k+=1
            elif i == j: #Fills the diagonal elements
                k = 0
                while k < N:
                    Inertia[i][j] += Mass[k] * (np.dot(Positions[k], Positions[k]) - np.dot(Positions[k], CartesianBasis[j])**2)
                    k+=1
    return Inertia




def Torque(Delta, ForceInput):
    Output = np.zeros((1,3))
    Output = np.cross(np.array(Delta), np.array(ForceInput))
    return Output

def NetTorque(Delta, RadiusInput, PositionInput, Index): #delta needs to be positions relative to COM
    #Position input needs to be absolute position
    Output = np.zeros((1,3))
    for i in range(N):
        Output += Torque(Delta, Flin(PositionInput, RadiusInput))
    return Output



#Quaternions/keeping track of rotations
def QuatMultiply(Quat1, Quat2):
    a1, b1, c1, d1 = Quat1
    a2, b2, c2, d2 = Quat2
    return np.array([a1*a2-b1*b2-c1*c2-d1*d2,
                    a1*b2+b2*a1+c1*d2-d1*c2,
                    a1*c2-b1*d2+c1*a2+d1*b2,
                    a1*d2+b1*c2-c1*b2+d1*a2])
#Keep track of orientation using quaternions
def VectorToQuat(VectorInput):
    VectorInput = np.array(VectorInput)#np.flip(np.array(VectorInput)) #I think it goes in the opposite direction (maybe add - signs in relevant places)
    #hopefully flipping order here takes care of the axis problems, so we get intended rotationmatrix back?
    QuatOutput = np.array([0,0,0,0])
    #print("np.shape 1x3",np.shape(VectorInput)[0])
    for i in range((np.shape(VectorInput)[0])):
        QuatOutput[i+1] = VectorInput[i]
    return np.array(QuatOutput)
#maybe flip order of elements so that the correct axis will rotate







#Center of mass motion

def COMVelocity(ForceInput, VelocityInput):
    VelocityInput = np.array(VelocityInput)
    Acceleration = np.array(ForceInput)/TotalMass
    VelocityOutput = VelocityInput + Acceleration*TimeStep
    return VelocityOutput

def COMPositionUpdate(ForceInput, PositionInput, VelocityInput): #Done in lab frame; no conversions needed
    PositionInput = np.array(PositionInput)
    #Add drag
    Acceleration = Drag(VelocityInput)/TotalMass
    VelocityInput = VelocityInput + COMVelocity(ForceInput, VelocityInput) +np.array(Acceleration)*TimeStep
    PositionOutput = PositionInput + VelocityInput*TimeStep
    return PositionOutput

def DeltaCOM(PositionInput1, PositionInput2):
    Delta = PositionInput1 - PositionInput2
    #New COM - Old COM; add these differences back onto HC at the end of simulation to get full motion
    return Delta

def DeltaPosition(PositionInput):
    Delta = PositionInput - COM
    return Delta

def LinDrag(PositionInput, OmegaInput): #v = w x r
    return -DragCoef*np.cross(PositionInput, OmegaInput)
