#this file generates the data structures used for the sim, fills, and runs the sim

#import src
import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as rot
import rowan as quat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


from src.config import Mass,Position,Radius,N,TotalMass,CartesianBasis,TimeStep,TotalSteps,g
from src import levitation_forces as l_f
from src.levitation_viz import DataFill,DataFill2
from src import levitation_math as l_m

#import src/levitation_viz.py

#if debug info will be printed
debug = True


#Check proper initialization:
if len(Position) != len(Mass) or len(Mass) != len(Radius) or len(Position) != len(Radius):
    #QUESTION perhaps initlaizationn should be moved to diff file?
    print("Initialization Error")
    exit(0)
else:
    print("Program Ready")
print("Number of Particles in System: \n", N)


#Calculates the center of Mass
WeightedPosition = np.zeros((N, 3))
COM = np.zeros((1,3))
print("Weighted Position (Zero): \n", WeightedPosition)
print("Center of Mass (Zero): \n", COM)
print("Sample Mass-Position Product: \n", Mass[1] * np.array(Position[1]))
i = 0
while i<N:
    WeightedPosition[i] = Mass[i] * np.array(Position[i])
    i +=1
COM = np.array((1/TotalMass) * sum(np.array(WeightedPosition)))


#Calcualtes the hydrodymanic Center - center weighted by surface area instead of mass
TotalArea = (np.linalg.norm(Radius))**2
WeightedPosition = np.zeros((N, 3))
for i in range(N):
    WeightedPosition[i] = (Radius[i]**2) * np.array(Position[i])
    i+=1
HydrodymanicCenter = np.array((1/TotalArea) * sum(np.array(WeightedPosition)))

#Redefine initial positions with respect to COM and HC
Position_RelativeToCOM, Position_RelativeToHC  = np.zeros((N,3)),np.zeros((N,3))
print("Position Relative to COM (Zeros): \n", Position_RelativeToCOM)
i = 0
while i<N:
    Position_RelativeToCOM[i] = Position[i] - COM
    Position_RelativeToHC[i] = Position[i] - HydrodymanicCenter
    i+=1

Inertia = l_m.InertiaTensorFill(Position_RelativeToCOM)

#Existence of 3 principal axes is always guaranteed (eigenvectors of Interia)
Eigen_Results = la.eig(Inertia) #Output of this function is a tuple (eigenvalues, eigenvectors)
PrincipalBasis = Eigen_Results[1] #Select the matrix of eigenvectors; it is a rotation matrix of the proper axes
PrincipalBasisInverse = np.linalg.inv(PrincipalBasis)
InertiaEigenvalues = np.array(Eigen_Results[0])

#Makes sure there are always three eigenvalues (fills with zeros, so we can dismiss those products in Euler equations)
if len(InertiaEigenvalues) != 3:
    BlankArray = [0,0,0]
    for j in range(len(InertiaEigenvalues)):
        BlankArray[j] = InertiaEigenvalues[j]
    InertiaEigenvalues = BlankArray

NumberofMoments = 0
for i in range(len(InertiaEigenvalues)):
    if InertiaEigenvalues[i] !=0:
        NumberofMoments +=1

#Diagonal Inertia (put 3 principal moments along the diagonal; invert)
DiagonalInertia = np.zeros((3,3))
for i in range(3):
    DiagonalInertia[i][i] = InertiaEigenvalues[i]

DiagonalInertiaInverse = np.zeros((3,3))
for i in range(3):
    if DiagonalInertia[i][i] !=0:
        DiagonalInertiaInverse[i][i] = 1/DiagonalInertia[i][i]


#Frame Changes (take only single array inputs):
def ToBodyFrame(Input):
    return np.dot(PrincipalBasis, Input)

def ToLabFrame(Input):
    return np.dot(PrincipalBasisInverse, Input)

def ToCOM(Input):
    return Input - COM

def ToOrigin(Input):
    return Input + COM

def ObtainQuatDot(OmegaInput, QuatInput):
    OmegaInput = l_m.VectorToQuat(ToBodyFrame(OmegaInput))
    #print("Omega Input",OmegaInput, "Quat Input", QuatInput)
    OutputQuat =  (0.5*l_m.QuatMultiply(np.array(QuatInput),np.array(OmegaInput)))
    #print("Output Quat:\n", OutputQuat)
    return OutputQuat #maybe change to cross product or write in matrix form

def RungeKuttaOrient(OmegaInput, QuatInput):
    #Make sure Omega is in body frame before putting it in here
    #Input matrix to quaternion
    QuatInput = rot.from_matrix(QuatInput)
    QuatInput = QuatInput.as_quat()

    #Does everything in body frame
    k1 = np.array( TimeStep * ObtainQuatDot(OmegaInput, QuatInput)  )
    k2 = np.array(TimeStep * ObtainQuatDot(OmegaInput, QuatInput + 0.5 * k1) )
    k3 = np.array(TimeStep * ObtainQuatDot(OmegaInput, QuatInput + 0.5 * k2) )
    k4 = np.array( TimeStep * ObtainQuatDot(OmegaInput, QuatInput + k3) )

    # Update next value of Omega
    QuatOutput = QuatInput + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

    #Syntactical acrobatics to return the correct rotation matrix
    #It would otheriwse have rotations around z and x axis confused
    QuatOutput = rot.from_quat(QuatOutput)
    #print("Quat Output (as quaternion);\n", QuatOutput.as_quat())
    #These conversions cause errors, but we need to flip the matrices around so things rotate around the axes properly --> need new funciton
    #QuatOutput = QuatOutput.as_euler("zyx")
    #QuatOutput = rot.from_euler("xyz", QuatOutput)
    QuatOutput = QuatOutput.as_matrix()
    #print("Quat Output (as matrix):\n", QuatOutput)
    #print("Norm: ", quat.norm(QuatOutput))
    QuatOutput = quat.normalize(QuatOutput)

    return QuatOutput #Gets rid of extra parentheses; in matrix form


def ObtbainOmegaDot(TorqueInput, OmegaInput): #Timestep 3rd argument
    #print("(ObtainOmegaDot) OriginalOmega:", OmegaInput)
    OmegaInput = ToBodyFrame(np.array(OmegaInput))  #Needs to be in fixed axis frame
    #print("(ObtainOmegaDot) Omega:", OmegaInput)
    TorqueInput = ToBodyFrame(np.array(TorqueInput))
    #print("Body Frame Torque:\n", TorqueInput)
    OmegaDot = np.dot(DiagonalInertiaInverse, TorqueInput + np.cross(np.dot(DiagonalInertia, OmegaInput), OmegaInput)) #change back to +
    return OmegaDot

def RungeKuttaSolver(TorqueInput, OmegaInput): #remember to put things in this order later in simulation
    #Iterate for number of iterations
    #print("(RungeKuttaSolver) Omega:", OmegaInput)
    Omega = ToBodyFrame(OmegaInput)
    #Does everything in body frame
    k1 = TimeStep * ObtbainOmegaDot(TorqueInput, Omega)
    k2 = TimeStep * ObtbainOmegaDot(TorqueInput, Omega + 0.5 * k1)
    k3 = TimeStep * ObtbainOmegaDot(TorqueInput, Omega + 0.5 * k2)
    k4 = TimeStep * ObtbainOmegaDot(TorqueInput, Omega + k3)

    # Update next value of Omega
    Omega = Omega + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    #Omega = Omega + TimeStep *ObtbainOmegaDot(TorqueInput, OmegaInput)
    #print("Omega:", Omega)

    return Omega

#Updates (1) particle positions
#        (2) HC position
# I don't think ToCOM is called anywhere, so it assumes all positions are with respect to [0,0,0] already
def PositionUpdate(OmegaInput, PositionInput):
    OmegaInput = ToLabFrame(np.array(OmegaInput))
    PositionInput = ToBodyFrame(ToCOM(np.array(PositionInput)))
    VelocityOutput = np.zeros((N,3))
    PositionOutput = np.zeros((N,3))
    VelocityOutput = np.cross(OmegaInput, PositionInput)
    PositionOutput = ToOrigin(ToLabFrame(PositionInput + VelocityOutput *TimeStep))
    return PositionOutput

def ObtainVelocity(OmegaInput, HCInput):
    OmegaInput = ToLabFrame(np.array(OmegaInput))
    VelocityOutput = np.cross(OmegaInput, HCInput) #Drag() was applied here
    return VelocityOutput

def HCPositionUpdate(VelocityInput, HCInput):
    #OmegaInput = ToLabFrame(np.array(OmegaInput))
    HCInput = ToBodyFrame(ToCOM(np.array(HCInput))) #Should this be in ToCOM() first
    #VelocityOutput = Drag(np.cross(OmegaInput, HCInput))
    #VelocityOutput = ObtainVelocity(OmegaInput, HCInput)
    #Add drag here (for now ... transfer to the PositionUpdate code above so tha we can make it proportional to the areas):
    #re-adjusts output velocity to be take into account drag that acts at this velocity
    #Acceleration = Drag(VelocityOutput)/TotalMass
    #DragAdjustment = np.array(Acceleration)*TimeStep
    #if np.dot(DragAdjustment, VelocityOutput) > 0: #ensures Drag and v are always opposite direction
    #    DragAdjustment = np.array([0,0,0])
    #VelocityOutput = VelocityOutput + np.array(Acceleration)*TimeStep
    #Final output
    HCOutput = ToOrigin(ToLabFrame(HCInput + VelocityInput *TimeStep)) #Switch order of ToLabFrame(ToOrigin()) --> do so in PositionUpdate() too!!
    return HCOutput

if(debug):
    print("Body Frame Torque:\n",ToBodyFrame(l_m.Torque([1,0,0], [0,1,0])))
    print("Position:\n",Position)
    print("Cartesian Basis:\n", CartesianBasis)
    Product = np.dot(Position[0], CartesianBasis)
    print("Matrix-Vector Product:\n",Product)
    Product = np.dot(CartesianBasis, Product)
    print("Matrix-Vector Product 2:\n",Product)
    print("len(Product)",len(Product))
    print("Cartesian Basis[1]:\n", CartesianBasis[1])
    print("Length Cartesian Basis",len(CartesianBasis)) #len() will count rows in the numpy matrix
    print("Center of Mass: \n", COM)
    print("Hydrodynamic Center: \n", HydrodymanicCenter)
    print("Actual Position Relative to COM: \n", Position_RelativeToCOM)
    print("Position Relative to Hydrodymanic Center: \n", Position_RelativeToHC)
    print("Length Position Relative to COM:\n", len(Position_RelativeToCOM)) #again, number of rows (i.e N)
    #Inertia
    print("Moment of Inertia Matrix:\n ", Inertia)
    print("Inertia Eigenvalues", InertiaEigenvalues)
    print("principal Basis (Unitary)\n", PrincipalBasis)
    print("Number of moments of inertia", len(Eigen_Results[0]))
    print("Number of Moments: ", NumberofMoments)
    print("Diagonal Inertia:\n", DiagonalInertia)
    print("Diagonal Inertia Inverse:\n ", DiagonalInertiaInverse)
    #Frame changes
    print("To Body COM Frame", ToBodyFrame(ToCOM(Position[0])))
    print("Original Position[0]", ToOrigin(ToLabFrame(ToBodyFrame(ToCOM(Position[0]))))) #note correct order of frame changes
    #Solve for OmegaDot and Omega
    Omega = [0,0,1]
    #Torque = [0,0,1] #gets confused with "Torque" function later
    #Rotation with quaternions
    print("Omega as Quat:\n", l_m.VectorToQuat(Omega))
    print("New Principal Basis:\n",RungeKuttaOrient(Omega, PrincipalBasis))
    print("Multiply:\n", l_m.QuatMultiply(l_m.VectorToQuat(Omega), np.array([1,2,3,4])))
    print("Alternate Multiply:\n", quat.multiply(l_m.VectorToQuat(Omega), np.array([1,2,3,4])))


#actual sim stuff
Step = 0


#QUESTION all this initializatios stuff should be in a function
#Initial Conditions; data to be stored through simulation
InitialOmega = np.array([0,0,0]) #This seems to be a good inital condition so that things don't weirdly drift off in y direction (about 10^-9m scale)
#[0,0,0.01] was used because in an earlier version of the simulation, they would just annihilate
AllOmegas = np.zeros((TotalSteps+1,3))
AllOmegas[0] = InitialOmega
#print("All Omegas:\n", AllOmegas)
AllHCPositions = []
AllHCPositions.append(1*HydrodymanicCenter)
AllCOMVelocities = np.zeros((TotalSteps+1, 3))
AllCOMPositions = np.zeros((TotalSteps+1, 3))
AllCOMPositions[0] = COM
AllDeltaCOM = np.zeros((TotalSteps+1, 3))
DragAdjustment = np.array([0,0,0]) #for the first iteration
Times = np.zeros((TotalSteps+1,1))
#AllParticlePositions = np.zeros((N, 3, TotalSteps+1))
#AllParticlePositions[0] = Position
Particles = np.zeros((N, TotalSteps+1, 3))
curr_vector = np.array([1, 0, 0]) #holds the rotation state to be printed
rot_vector_list = []
euler_angle_list = []
for i in range(N):
    Particles[i][0] = Position[i]
print("Particles\n", Particles)
print("Particles[0]\n", Particles[0])
print("Particles[0][0]\n", Particles[0][0])
print("Particles[1][0]\n", Particles[1][0])
print("Particles[1][0][0]:\n", Particles[1][0][0])


print("All HC:\n", AllHCPositions)
#COM = [0,0,0] #SOMETHING WRONG with change of frame mechanism that's making things diverge far beyond any length scale of the problem
#TotalSteps = 2 #Just for testing
while Step < TotalSteps:
    #print("Principal Basis: ", Step, PrincipalBasis)
    #Solve for torques
    #Add LeverArm, Torque calcultion when confident that simple torque works
    #It does work; so LeverArm, Torque should be fine; continute using TorqueZ for testing
    TorqueZ = np.array(l_m.Torque(np.array(ToCOM(AllHCPositions[Step])), (np.array([0,0,-g*TotalMass]))))
    #print("torquez is ",TorqueZ)

#calculates torque based on gravity and old height



                                #Drag adjustment
    ForceZ = [0,0,-0.01]
    #AllOmegas[Step] = np.dot(np.linalg.inv(PrincipalBasis), AllOmegas[Step])
    AllOmegas[Step+1] = RungeKuttaSolver(TorqueZ, AllOmegas[Step]) #pay attention to argument order!!!
    #DragAdjustment = LinDrag(AllHCPositions[Step], AllOmegas[Step]) #Apply this linear drag (maybe use AllOmegas[Step]?)
    #Don't adjust Torque but only just readjust the position based on the DragAdjustment vector?????

    #Update HC positions
    #AllOmegas[Step] = np.array(AllOmegas[Step]).T
    #Need to somehow take AllOmega[Step+1] out of below so we can apply Drag() in a separate line, then feed a velocity back to the code line below
    Velocity = np.array([0,0,0])
    Velocity = l_f.Drag(ObtainVelocity(AllOmegas[Step+1], AllHCPositions[Step])) #Applying drag

#this is angular velocity rigth??

    #what velocity is this
    AllHCPositions.append(HCPositionUpdate(Velocity, AllHCPositions[Step])) #It works with AllOmegas[Step+1] for sure

    #Supposedly it works??
    #Update particle positions
    for k in range(N):
        Particles[k][Step+1] = PositionUpdate(AllOmegas[Step+1], Particles[k][Step])

    #Update principal basis
    #print("Principal Basis:\n", PrincipalBasis)
    PrincipalBasis = RungeKuttaOrient(AllOmegas[Step+1], PrincipalBasis)
    PrincipalBasis = np.array(PrincipalBasis)
    #orientations seem to creep in unwanted positions (actual expected movement is on the order of 10^-5)
    PrincipalBasisInverse = np.linalg.inv(PrincipalBasis)
#this keeps track of rotation I think
    #print("Principle Basis inverse:\n",PrincipalBasisInverse)
    rotation_object = rot.from_matrix(PrincipalBasisInverse)
    #print(PrincipalBasisInverse)

    curr_vector = rotation_object.apply(curr_vector)
    #print(curr_vector)
    rot_vector_list.append(curr_vector.copy())
    #euler_angle_list.append (rotation_object.as_euler('xyz'))


    AllCOMVelocities[Step+1] = l_m.COMVelocity(ForceZ, AllCOMVelocities[Step])
    COM = l_m.COMPositionUpdate(ForceZ, COM, AllCOMVelocities[Step+1]) #Do not condense these steps because COM itself is called in in each iteration!!!
    AllCOMPositions[Step+1] = l_m.COMPositionUpdate(ForceZ, COM, AllCOMVelocities[Step+1])

    #Adjust for COM motion
    AllDeltaCOM[Step+1] = AllCOMPositions[Step+1] - AllCOMPositions[Step]
    AllHCPositions[Step+1] = AllHCPositions[Step+1] + AllDeltaCOM[Step+1]
    for k in range(N):
        Particles[k][Step+1] = Particles[k][Step+1] + AllDeltaCOM[Step+1]

    #Drag implementation:
    #Obtain particle/HC velocity, calculate drag adjustment, feed back into torque at top of loop

    if Step%1000 == 0:
        print("Simulation Step", Step)

    Times[Step+1] = Times[Step] + TimeStep
    Step +=1

###################################################################################################################
#print("All Omegas", AllOmegas)


#here I'll be visualizing the rotation
for vector in rot_vector_list:
    print(vector)

print("About to begin plottinng vectors")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def get_arrow(i):
    x = 0
    y = 0
    z = 0
    u = rot_vector_list[i][0]
    v = rot_vector_list[i][1]
    w = rot_vector_list[i][2]
    print(rot_vector_list[i], i)
    return x,y,z,u,v,w

quiver = ax.quiver(*get_arrow(0))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

def update(theta):
    global quiver
    quiver.remove()
    quiver = ax.quiver(*get_arrow(theta))

ani = FuncAnimation(fig, update, frames=range(4000), interval=50)
plt.show()





#Plot Data
print("Number of HC Positions: ", len(AllHCPositions))
#Plotting capabilities
fig = plt.figure()
ax = plt.axes(projection='3d')





# Data for three-dimensional scattered points: HC Positions
xdata = DataFill(AllHCPositions, 0)
ydata = DataFill(AllHCPositions, 1)
zdata = DataFill(AllHCPositions, 2)
xdata0 = DataFill(AllCOMPositions, 0)
ydata0 = DataFill(AllCOMPositions, 1)
zdata0 = DataFill(AllCOMPositions, 2)
#Need to fill by column and "layer" because all positions are stored in one array
#Middle "0" is the first particle
xdata1 = DataFill2(Particles, 0, 0)
ydata1 = DataFill2(Particles, 0, 1)
zdata1 = DataFill2(Particles, 0, 2)
#Second particle
xdata2 = DataFill2(Particles, 1, 0)
ydata2 = DataFill2(Particles, 1, 1)
zdata2 = DataFill2(Particles, 1, 2)
#Fill further particles if needed ...

ax.scatter3D(xdata, ydata, zdata, color = "Blue")
ax.scatter3D(xdata0, ydata0, zdata0, color = "Orange")
#ax.scatter3D(xdata1, ydata1, zdata1, color = "Purple")
#ax.scatter3D(xdata2, ydata2, zdata2, color = "Purple")
plt.show()
