#!/usr/bin/env python3
#%%
import numpy as np
from typing import Tuple
from numpy.random import default_rng
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib as mpl
import sys
from scipy.integrate import solve_ivp
import warnings
#mpl.rcParams['figure.dpi'] = 200
BOLTZMANN = 1.380649e-23 # k_B Boltzmann constant, units of J/K
IDEALGAS = 8.314 # for verification, units J / (K mol)
AVOGADRO = 6.023e23 # Avogadro's number of things per mole
COLLISION_TIME = 1e-9

# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]
#%%

class ParticleSimulator:

    def __init__(self, cuberoot_N: int = 3, temperature = 50, scenario: str = 'ideal', seed = 2):
        '''
            Initializes the particle simulation

            Param:
                cuberoot_N: the cube root number of particles in the simulation
                temperature: degrees Kelvin
        '''
        self.N = cuberoot_N**3
        rng = default_rng(seed=seed)

        if scenario == 'ideal':
            self.MASS = 1e-20 # in kg
            # Potential constants
            self.V0 = 1e-9
            self.A = 1e-5 # magic numbers from Elio's desmos
            self.MIN_SEPARATION = 9.99e-6
        elif scenario == 'nonideal':
            self.MASS = 1e-20
            self.V0 = 1e-6
            self.A = 1e-5
            self.MIN_SEPARATION = self.A * 0.999999999999 # 12 9s
        elif scenario == 'nonideal2':
            self.MASS = 1e-20
            self.V0 = 1e-8
            self.A = 1e-5
            self.MIN_SEPARATION = self.A * 0.999999999999 # 12 9s
        else:
            raise f"Unknown scenario {scenario}"
    
        self.scenario = scenario


        # generate starting positions in spherical coordinates so that we stay withing our bounds
        spherCoord = rng.uniform([0, 0, 0], [1, np.pi, 2 * np.pi], size=(self.N, 3))

        # generate particles on a grid in a cube of side length 1.4 inside the sphere
        self.posX, self.posY, self.posZ = np.meshgrid(
            np.linspace(-0.5, 0.5, cuberoot_N),
            np.linspace(-0.5, 0.5, cuberoot_N),
            np.linspace(-0.5, 0.5, cuberoot_N)
        )

        self.temp = temperature
        self.posX = self.posX.ravel()
        self.posY = self.posY.ravel()
        self.posZ = self.posZ.ravel()

        initial_velocity = np.sqrt(3 * BOLTZMANN * temperature / self.MASS)
        print(f"Initial velocity for T = {temperature} K is {initial_velocity} m/s.")
        # generate velocity with constant magnitude v_initial
        # asking numpy RNG to give a number between v_i and v_i each time should yield v_i 
        velocities = self._sphericalToCart(rng.uniform(
            [initial_velocity, 0, 0],
            [initial_velocity, np.pi, 2 * np.pi],  size=(self.N, 3)
        ))
        self.velX = velocities[:, 0]
        self.velY = velocities[:, 1]
        self.velZ = velocities[:, 2]
        



        # self.accel = np.zeros((3, self.N))
  
    def checkCollisionsWithSphere(self, pos, vel):
        """Redirects particles which are outside of the unit sphere"""
        # pos = np.array([pos])
        norms = np.linalg.norm(pos, axis=1) 
        # indices of all particles outside the unit sphere
        outside_indices = np.where(norms > 1)
# 
        # outward normal vector of the sphere which we will use to find the tangent plane
        normal_vector = pos[outside_indices]
        #normalize it
        normal_vector = normal_vector / np.linalg.norm(normal_vector, axis=1, keepdims=True)
    
        # # if nothing is outside then return
        if np.size(outside_indices) == 0:
            return pos, vel
        # subtract twice the projection of the velocity onto the tangent plane of the sphere
        # this reflects the change in velocity due to elastic collision with the inside of the sphere
    #https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    
    #https://stackoverflow.com/questions/68245372/how-to-multiply-each-row-in-matrix-by-its-scalar-in-numpy
        #print('norm',normal_vector)
        #teleport back into sphere if outside of sphere  
        # print("MADE IT", outside_indices, pos)  
        pos[outside_indices] = normal_vector #[outside_indices] = normal_vector
        #change the velocity from collision
        
        
        #print((vel[outside_indices] @ np.transpose(normal_vector)).diagonal()[:,None])
        # print(np.sum(self.vel[outside_indices] @ np.transpose(normal_vector), axis=1)[:, None])
        projected_velocity = (vel[outside_indices] @ np.transpose(normal_vector)).diagonal()[:,None] * normal_vector
        vel[outside_indices] -= 2 * projected_velocity
        magnitude_projected_velocity = np.linalg.norm(projected_velocity, axis=1)

        # force = dP/dt
        total_impulse_exerted = np.sum(2 * self.MASS * magnitude_projected_velocity)
        #print('vel',projected_velocity)
        return pos, vel, total_impulse_exerted
        

    def kineticE(self) -> float:
        """Kinetic energy"""
        x = np.sum(0.5 * self.MASS * np.square(np.linalg.norm(self.vel, axis = 1)))
        # print(x)
        return x
    
    @staticmethod
    def _sphericalToCart(coord):
        '''
            Converts spherical coordinates to cartesion... we think
        '''
        r, theta, phi = coord[:,0], coord[:,1], coord[:,2]
        return np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ]).transpose()
    
    def force(self, a: Coordinate, b: Coordinate) -> Coordinate:
        """
            Returns the force caused by the particle at position a
            for the particle at position b, as an x-y-z tuple.

            This is based on the potential function V(r) = V_0 * ((a/r)^4 - (a/r)^3) per Prof. Saeta.
            (where r is the distance between the particles and a is some arbitrary constant)

            We can write this in Cartesian coordinates as V(x, y, z) = 
                V_0 (a^4/(x^2 + y^2 + z^2)^2 - a^3/(x^2+y^2+z^2)^(3/2))
            Per WolframAlpha we have the following components of F(x, y, z) = -grad(V):
                * (x direction) = -V_0(a^3 x (-(4 a)/(x^2 + y^2 + z^2)^3 + 3/(x^2 + y^2 + z^2)^(5/2))
                * (y direction) = -V_0 a^3 y (-(4 a)/(x^2 + y^2 + z^2)^3 + 3/(x^2 + y^2 + z^2)^(5/2))
                * (z direction) = -V_0 a^3 z (-(4 a)/(x^2 + y^2 + z^2)^3 + 3/(x^2 + y^2 + z^2)^(5/2))
        """

        # ensure we dont calculate the force of a particle on itself
        # print(f"a:\n{a}\nb:\n{b}")
        # if np.all(a == b):
        #     return 0
        # x,y,z components of r
        # diff = a - b
        # x = diff[0]
        # y = diff[1]
        # z = diff[2]

        radii = a - b
        x = radii[:,0]
        y = radii[:,1]
        z = radii[:,2]
        rsquare_real = x**2 + y**2 + z**2
        # find the zero radius index, which is the index of the force of the particle on itself
        # which is not a physical force and therefore will be set to zero later.
        zero_radius_idx = np.where(rsquare_real == 0)[0][0]
        # replace too-small separations by MIN_SEPARATION
        rsquare_corrected = np.where(
            rsquare_real < self.MIN_SEPARATION ** 2, self.MIN_SEPARATION ** 2, rsquare_real
        )

        F = np.array([
            -self.V0 * (self.A**3 * x * (-(4 * self.A)/(rsquare_corrected)**3 + 3/(rsquare_corrected)**(5/2))),
            -self.V0 * self.A**3 * y * (-(4 * self.A)/(rsquare_corrected)**3 + 3/(rsquare_corrected)**(5/2)),
            -self.V0 * self.A**3 * z * (-(4 * self.A)/(rsquare_corrected)**3 + 3/(rsquare_corrected)**(5/2))
        ])
        F[:,zero_radius_idx] = 0 # remove force of a particle on itself
        # An array has 
        # n = np.linalg.norm(F)
        # if n > 1:
        #     print(n)
        return F
    
    def runIVP(self, t, fps):

        # print(passedVals)
        passedVals = np.concatenate((
            self.posX,
            self.posY,
            self.posZ,
            self.velX,
            self.velY,
            self.velZ
        ))
        N = self.N

        # solve_ivp does not like us changing the velocity on collision
        # so we stop solving using an event when there is a collision
        def event(t, U):                
            pos = np.array([U[:N], U[N:2*N], U[2*N:3*N]]).T
            if np.any(np.linalg.norm(pos, axis=1) > 1):
                return 0.0 # event! we found a collision! terminate
            else:
                return -1.0 # keep going!
        event.terminal = True

        dataset = np.zeros((N * 6, fps * t ))
        frame = 0
        timeList = np.linspace(0, t, int(fps * t ))
        netImpulseOnSphere = np.zeros(len(timeList))
        propConst = np.zeros(len(timeList)) #PV / (nT)

        tPick = 0
        while frame < int(fps * t)-1:
            data = solve_ivp(
                self.dU, t_span=(tPick,t), y0=passedVals,
                t_eval = timeList[frame:],
                max_step = 0.001, events=event, dense_output=False,
                first_step = COLLISION_TIME,
                rtol=1e-6, atol = 1e-9 # 1000x more sensitive to error
                #min_step = 0.000001
            )

            
            # times = np.concatenate((times, data.t[1:]))
            # put in place
            frameGen = len(data.t) 
            
            dataset[:, frame: frame + frameGen] = data.y
            
            # update the frame
            frame +=frameGen
            if len(data.y_events[0]) != 0: #if there is a collision event

                thisInst = data.y_events[0][0]
                pos = np.array([thisInst[:N], thisInst[N:N*2], thisInst[N*2:N*3]]).T
                vel = np.array([thisInst[N*3:N*4], thisInst[4*N:N*5], thisInst[N*5:N*6]]).T
                pos, vel, impulse = self.checkCollisionsWithSphere(pos, vel)
                netImpulseOnSphere[frame] += impulse

                pos = pos.T
                vel = vel.T
                passedVals = np.concatenate((pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
                tPick = data.t_events[0][0]
                propConst[frame] = (np.sum(netImpulseOnSphere) / ( (frame/fps)* 3) ) / (self.N * self.temp / AVOGADRO)# * np.pi*4/3 / self.N / self.temp
                print(f'frame {frame}, constant: {propConst[frame]}')
                #np.sum(netImpulseOnSphere) / (4*np.pi * (frame/fps))
        #plot the sphere taken from matplotlib docs

        # the proportionality constant wont be filled in if no collisions during that frame, thus we can fill it in.
        blankSpots = np.where(propConst == 0.0)

        # fill in the blank spots with a loop....
        for i in blankSpots[0][1:]:
            propConst[i] = (np.sum(netImpulseOnSphere[:i]) / (timeList[i]*3)) / (self.N * self.temp / AVOGADRO)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
 
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
 
        print('data has been generated! yay!')
        print(f"Total impulse: {np.sum(netImpulseOnSphere)}, prop constant: {propConst[-1]}")
        print(f'Ideal Gas constant: {IDEALGAS}')

        fig = plt.figure(figsize=plt.figaspect(0.5))

        # Avogadro plot
        ax_prop = fig.add_subplot(2,2,2)
        prop = ax_prop.plot(timeList[:1], propConst[:1], label='Calculated constant')[0]
        ax_prop.set_ylim((np.min(propConst)-0.5, np.max(propConst)*1.1))
        ax_prop.set_xlabel('Time (seconds)')
        ax_prop.set_ylabel(r'$\frac{PV}{nT}$')
        ax_prop.set_title('Ideal gas constants')
        ax_prop.axhline(y=8.314, label='actual ideal gas constant', ls=':')
        ax_prop.legend()


        
        # particle plot
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(x, y, z, alpha=0.1)
       
        xp = dataset[:self.N,:].T
        yp = dataset[self.N:self.N*2,:].T
        zp = dataset[self.N*2:self.N*3,:].T

        xv = dataset[self.N*3:self.N*4,:].T
        yv = dataset[self.N*4:self.N*5,:].T
        zv = dataset[self.N*5:self.N*6,:].T

        KE = (np.square(xv) + np.square(yv) + np.square(zv)) * self.MASS / 2
        KEmin = np.min(KE)
        KEmax = np.max(KE)

        print('calc potential')
        pEtotal = np.zeros(len(timeList))
        for i in range(len(timeList)):
            pEtotal[i]= self.calcPotential(xp,yp,zp,i)

        #print(potential)

        
        scat = ax.scatter(xp[0], yp[0], zp[0], c = KE[0], cmap='cool', 
                          vmin=KEmin, vmax = KEmax)
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))
        ax.set_title(f'Particle simulation (T = {self.temp} K)')
        ax.set_aspect('equal')

        


        # energy plots

        kEtotal = np.sum(KE, axis=1)
        # emin = np.min(np.concatenate((kEtotal[1:-1], pEtotal)))
        # emax = np.max(np.concatenate((kEtotal[1:-1], pEtotal)))
        kmin = np.min(kEtotal[1:-1])
        kmax = np.max(kEtotal[1:-1])
        pmin = np.min(pEtotal)
        pmax = np.max(pEtotal)
        #https://matplotlib.org/stable/gallery/spines/multiple_yaxis_with_spines.html

        ax_KE = fig.add_subplot(2,2,4)
        ax_PE = ax_KE.twinx()
        #plot kinetic and potential
        kinetic = ax_KE.plot(timeList[:1], kEtotal[:1], label=r'$K_E$', color = 'steelblue')[0]
        pot = ax_PE.plot(timeList[:1], pEtotal[:1], label = r'$|U_E|$', color='forestgreen')[0]


        #add labels
        ax_KE.set_xlabel('Time (seconds)')
        ax_KE.set_ylabel('$K_E$ (J)')
        ax_PE.set_ylabel('$|U_E|$ (J)')

    # set limits and such
        ax_KE.set_title('System energy')
        ax_KE.set_ylim((kmin*0.75, kmax*1.25))
        ax_PE.set_ylim((pmin*0.75, pmax*1.25))

        ax_KE.legend(handles = [kinetic, pot])

    # set colors
        ax_KE.tick_params(axis='y', colors=kinetic.get_color())
        ax_PE.tick_params(axis='y', colors=pot.get_color())



        rotateRate = 5 / (fps)
        def update(frame):
            # particles
            scat._offsets3d = (xp[frame], yp[frame], zp[frame])
            scat.set_array(KE[frame])
            scat.set_clim(vmin=KEmin, vmax=KEmax)

            # Avogadro
            prop.set_xdata(timeList[:frame])
            prop.set_ydata(propConst[:frame])
            ax_prop.set_xlim((0, timeList[frame]+0.1))

            kinetic.set_xdata(timeList[:frame])
            kinetic.set_ydata(kEtotal[:frame])
            pot.set_xdata(timeList[:frame])
            pot.set_ydata(pEtotal[:frame])
            ax_KE.set_xlim((0, timeList[frame]+0.1))
            # REMOVE IF PLOTTING LIVE
            #ax.view_init(30, 60 + rotateRate * frame, 0)
            return (scat, prop, kinetic, pot)
        
        fig.tight_layout(pad=.5)

        ani = animation.FuncAnimation(fig = fig, func = update, frames = t * fps, interval = (1000/fps) - 1)
        # https://stackoverflow.com/questions/37146420/saving-matplotlib-animation-as-mp4
        #ani.save(f'Particles-{self.scenario}-{self.temp}.mp4', writer = animation.FFMpegWriter(fps=fps))

        plt.show()

    def potential(self, a: Coordinate, b: Coordinate) -> Coordinate:
        '''potential function: V(x, y, z) = 
                V_0 (a^4/(x^2 + y^2 + z^2)^2 - a^3/(x^2+y^2+z^2)^(3/2))'''

        # ensure we dont calculate the force of a particle on itself
        # print(f"a:\n{a}\nb:\n{b}")
        # if np.all(a == b):
        #     return 0
        # x,y,z components of r
        # diff = a - b
        # x = diff[0]
        # y = diff[1]
        # z = diff[2]

        radii = a - b
        x = radii[:,0]
        y = radii[:,1]
        z = radii[:,2]
        rsquare = x**2 + y**2 + z**2
        with warnings.catch_warnings():
            #suppress divide by 0 warnings
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            potential = self.V0 *  (self.A**4 / rsquare**2) - self.V0 * (self.A**3 / rsquare**(3/2))

        #get rid of nan vals from calculating potential with self
        np.nan_to_num(potential, False, nan=0.0)
        return potential


    def calcPotential(self, xp, yp, zp, frame):
        '''potential function: V(x, y, z) = 
                V_0 (a^4/(x^2 + y^2 + z^2)^2 - a^3/(x^2+y^2+z^2)^(3/2))'''
        
        pos = np.array([xp[frame], yp[frame], zp[frame]]).T
        potentialList = np.apply_along_axis(
            lambda r: np.sum(self.potential(pos, np.tile(r, (self.N,1))), axis = 0),
            axis=1,
            arr=pos
        )
        
        return np.abs(np.sum(potentialList)/2)

    
    def dU(self, t, U):
        '''
            returns the derivative of position and velocity 
            args:
                U unpacks to: x, y, z, vx, vy, vz
            returns:
                vx, vy, vz, ax, ay, az
        '''
        N = self.N
        # unwraps the conditions into a position matrix and a velocity matrix.
        pos = np.array([U[:N], U[N:2*N], U[2*N:3*N]]).T
        vel = np.array([U[3*N:4*N], U[4*N:5*N], U[5*N:6*N]])
        
        
        # print("vel", vel)

        # makes a matrix for each particle position, calculates all the forces 
        # on it and sums, and divides by mass to get accel
        newAccel = np.apply_along_axis(
            lambda r: np.sum(self.force(pos, np.tile(r, (self.N,1))), axis = 1) / self.MASS,
            axis=1,
            arr=pos
        )

        # re wraps the velocity and acceloration to be passed back
        newAccel = newAccel.T

        return np.concatenate((vel[0], vel[1], vel[2], newAccel[0], newAccel[1], newAccel[2]))

s = ParticleSimulator(scenario='nonideal2')
print('Simulation Initiated')
#s.run()
# s.runPre(0.01, 5)
s.runIVP(2, 40)
# %%
