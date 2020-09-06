import numpy as np
from src.config import TotalMass,g,DragCoefficient,TimeStep
#Let's take the force field to be something simple: F = k (0,0,1/z) so that k makes the units work out (Set k = 1)
#Force = [0,0,0] #Initialize force outide definition so we can call it


def F(PositionInput):
    if np.dot(PositionInput, CartesianBasis[2]) != 0:
        Component = 1/(np.dot(PositionInput, CartesianBasis[2]))
        Component = np.abs(Component)
    else: #deals with discontinuity
        Component = 1
    Force = [0,0,Component]
    return Force

#Drag Force Quadratic (Reynolds number ~1.5): Fdrag = -Kv (in the v-hat direction)
#Linear holds generally if Re < 1
#NOT CALLED YET in the position and rotation functions
#QUESTION will be called seperately for each particle, correct
def Drag(VelocityInput): #Make it linear for small velocities
    #Quadratic Drag
    VelocityMagnitude2 = np.linalg.norm(np.array(VelocityInput))
    if VelocityMagnitude2 == 0:
        VelocityDirection = np.zeros((1,3))
    else:
        VelocityDirection = VelocityInput/VelocityMagnitude2
    #print(VelocityMagnitude2, VelocityDirection)
    Adjustment = - (DragCoefficient / TotalMass) *VelocityInput*TimeStep
    DragVelocity = VelocityInput  + Adjustment #VelocityMagnitude2 * np.array(VelocityDirection)
    if np.dot(Adjustment, VelocityInput) > 0:
        DragVelocity = np.array([0,0,0])
    return DragVelocity

#Translational force on the body (effectively acts on COM)
def NetForce(PositionInput, RadiusInput, Delta, Index): #delta input should be direction from COM to particle in question
    #This component of the force is parallel to r; perpendicular to direction that induces torque
    Delta = Delta/(np.linalg.norm(Delta)) #unit vector direction
    for i in range(N): #sum forces induced by each particle in "Particles"
        FOut += np.dot(Flin(PositionInput[i][Index], RadiusInput[i]), Delta) #component of force in unit direction #Index = Step
    FOut = FOut + Fg #Sum of translational forces
    return FOut #Call in simulation while loop


#Alternate force that is continuous everywhere (may be more useful as a trial function later)
#are these even being used
ThermCoef = 30
Fexp = lambda PositionInput: [0,0,50*np.exp(-(np.dot(PositionInput, CartesianBasis[2])**2))]
Flin = lambda PositionInput, RadiusInput: np.array([0,0,-ThermCoef *(RadiusInput**2) * np.dot(PositionInput, CartesianBasis[2])])
#Force proportional to r^2 for each particle

#Gravitational Force
Fg = TotalMass* np.array([0,0,-g])
