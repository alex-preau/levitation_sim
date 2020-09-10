import numpy as np
from src.config import TotalMass,g,DragCoefficient,TimeStep
from scipy import optimize
import matplotlib.pyplot as plt
#Let's take the force field to be something simple: F = k (0,0,1/z) so that k makes the units work out (Set k = 1)
#Force = [0,0,0] #Initialize force outide definition so we can call it

grad = [-37846.106761534735, -32243.056428960837, -28595.288333315522, -25971.927269235486, -23964.234676178115, -22360.58696188197, -21038.92162971438, -19923.215990742872, -18963.31582260778, -18124.615443352788, -17382.363665073783, -16718.328490688633, -16118.746143383692, -15573.007957629301, -15072.790573608861, -14611.462757447565, -14183.670552181486, -13785.04069002915, -13411.964399519202, -13061.437076457394, -12730.93753697917, -12418.335809435805, -12121.821827376478, -11839.84964701455, -11571.093342892447, -11314.411789324176, -11068.820272675688, -10833.467403245268, -10607.616172515443, -10390.62827646477, -10181.95102852399, -9981.106337218194, -9787.68133789732, -9601.320355190239, -9421.717940101777, -9248.612778084836, -9081.782305658524, -8921.03790592262, -8766.220579593777, -8617.197009448193, -8473.855953350385, -8336.104915242657, -8203.867055103918, -8077.0783085024905, -7955.684694228203, -7839.639794953865, -7728.902401073311, -7623.434312030718, -7523.198292693464, -7428.156184749451, -7338.267174850476, -7253.486222338945, -7173.762649993161, -7099.038901343482, -7029.249467867894, -6964.319988766454, -6904.1665251554305, -6848.69500942526, -6797.800869223533]
temp = [77, 94.81160898039097, 109.98624020637028, 123.44411310420327, 135.6673469470287, 146.9456953450794, 157.46931496192897, 167.37091587952798, 176.74742942938815, 185.67218273445582, 194.2022163580217, 202.3829221491897, 210.25111138614784, 217.83711766597827, 225.16628180266375, 232.26002736629744, 239.13665705259618, 245.81195381096472, 252.29964238872694, 258.6117491279585, 264.7588863062611, 270.75047965009543, 276.594952451976, 282.29987612826386, 287.8720945238844, 293.31782746011015, 298.6427577073448, 303.85210459772304, 308.95068677169013, 313.942976009508, 318.83314368480205, 323.62510105891914, 328.3225343878057, 332.92893561987336, 337.4476293111338, 341.88179626336546, 346.23449429522014, 350.50867647973564, 354.70720712070073, 358.8328756916692, 362.88840892277244, 366.87648118997424, 370.7997233375847, 374.6607300465003, 378.4620658468121, 382.2062698632975, 385.89585937522986, 389.53333226730126, 393.1211684457796, 396.661830292868, 400.15776223222673, 403.6113894794267, 407.0251160524401, 410.4013221188902, 413.74236075845323, 417.05055422036395, 420.32818975724575, 423.5775151173528, 426.8007337776621]


Rc = .062364 #
mfpconstant = 428081 #the constant part of the denominator of calculating mean free path. Has units of m^2
kb = 1.38065 * 10**(-23) ## Boltzmann constant metres^2 * kg * seconds^(-2) * Kevlin^(-1)
pressure = 5 #torr
mN = 2.3*10**(-23) # mass of a nitrogen molecule
gas_constant = 287.05

grad = np.multiply(grad,-1)[::-1]
temp = temp[::-1]

#calculate Mean-Free-Path of air for a given temperature and pressure (Given in meters)
def mfp(T,P):
    return Rc*T/(mfpconstant*P)

#calculate thermophoretic constant given the Knudsen Number
def Fth(Kn):
    return 1.89 * ((Kn**2)-.115*Kn+.00611)/((Kn**2)+.5284*Kn+.6508)


#calculate thermal conductivity for a given temperature (given in mW/(m*K))
def thermc(T):
    return -0.00000000000000000375*T**(6)+0.00000000000001909230*T**(5)-0.00000000004016396062*T**(4)+0.00000004901175449123*T**(3) \
        -0.000044075614398888*T**(2)+0.0766069577308689*T+24.3560822452597

def get_density(temp):
    pascals = pressure * 133.32
    return pascals /(temp * gas_constant)

#calculates total force given a temperature, gradient, pressure, and radius
#I'm pretty sure this gives the force in Newtons
def ThermForce(T,grad,P,a):
   # print(-Fth(mfp(T,P)/a)*BasicForce(T,grad)*(a**2))
    return -Fth(mfp(T,P)/a)*BasicForce(T,grad)*(a**2)



#given a temperature and gradient, calculate the force without thermophoretic constant or radius (milliNewtons/m^2)
def BasicForce(T,grad):
    #T = calc_temp(height)
    return thermc(T)*grad/np.sqrt(2*kb*T/mN)
def exponential(p, xvar):
    return  p[0] * np.exp(xvar*p[1]) + p[2]

def quad(p,xvar):
    return p[0] + p[1]*xvar+ p[2]*np.power(xvar,2)

def quad_residual(p, xvar, yvar, err):
    return (quad(p, xvar) - yvar)/err

def exponential_residual(p, xvar, yvar, err):
    return (exponential(p, xvar) - yvar)/err
dy = np.full(59,.01)

print("About to fit force approx func")

p0 = [ 1.,1.,1.]
x = range(59)
pf, cov, info, mesg, success = optimize.leastsq(exponential_residual, p0,
                                                args=(x, grad, dy), full_output=1)

# If the fit failed, print the reason
if cov is None:
    print('Fit did not converge')
    print('Success code:', success)
    print(mesg)
else:
    chisq = sum(info['fvec']*info['fvec'])
    dof = len(x) - len(pf)
    pferr = [np.sqrt(cov[i,i]) for i in range(len(pf))]
    print('Converged with chi-squared', chisq)
    print('Number of degrees of freedom, dof =', dof)
    print('Reduced chi-squared ', chisq/dof)
    print('Inital guess values:')
    print('  p0 =', p0)
    print('Best fit values:')
    print('  pf =', pf)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, grad, dy, fmt='ko', label = 'Data')
X = np.linspace(0, 59, 500)
ax.plot(X, exponential(pf, X), 'r-', label = 'Linear Fit')

ax.set_title('Plot of therm force')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Here is the text we want to include...
textfit = '$f(x) = A + Bx$ \n' \
          '$A = %.2f \pm %.2f$ \n' \
          '$B = %.2f \pm %.2f$ \n' \
          '$\chi^2= %.1f$ \n' \
          '$N = %i$ (dof) \n' \
          '$\chi^2/N = % .2f$' \
           % (pf[0], pferr[0], pf[1], pferr[1],
              chisq, dof, chisq/dof)
#... and below is where we actually place it on the plot
#ax.text(0.05, 0.90, textfit, transform=ax.transAxes, fontsize=12,
#        verticalalignment='top')

ax.set_xlim([0-0.5, 59+0.5])
plt.show()

grad_approx_coef = pf
def calc_grad(x):
    return -(exponential(grad_approx_coef,x))/1000

p0 = [ 1.,1.,1.]
x = range(59)
pf, cov, info, mesg, success = optimize.leastsq(quad_residual, p0,
                                                args=(x, temp, dy), full_output=1)

# If the fit failed, print the reason
if cov is None:
    print('Fit did not converge')
    print('Success code:', success)
    print(mesg)
else:
    chisq = sum(info['fvec']*info['fvec'])
    dof = len(x) - len(pf)
    pferr = [np.sqrt(cov[i,i]) for i in range(len(pf))]
    print('Converged with chi-squared', chisq)
    print('Number of degrees of freedom, dof =', dof)
    print('Reduced chi-squared ', chisq/dof)
    print('Inital guess values:')
    print('  p0 =', p0)
    print('Best fit values:')
    print('  pf =', pf)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x, temp, dy, fmt='ko', label = 'Data')
X = np.linspace(0, 59, 500)
ax.plot(X, quad(pf, X), 'r-', label = 'Linear Fit')

ax.set_title('Some Sample Data with Error Bars')
ax.set_xlabel('x')
ax.set_ylabel('y')


# Here is the text we want to include...
textfit = '$f(x) = A + Bx$ \n' \
          '$A = %.2f \pm %.2f$ \n' \
          '$B = %.2f \pm %.2f$ \n' \
          '$\chi^2= %.1f$ \n' \
          '$N = %i$ (dof) \n' \
          '$\chi^2/N = % .2f$' \
           % (pf[0], pferr[0], pf[1], pferr[1],
              chisq, dof, chisq/dof)
#... and below is where we actually place it on the plot
#ax.text(0.05, 0.90, textfit, transform=ax.transAxes, fontsize=12,
#        verticalalignment='top')

#ax.set_xlim([0-0.5, 59+0.5])
plt.show()

temp_approx_coef = pf
def calc_temp(x):
    return (quad(temp_approx_coef,x))

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
