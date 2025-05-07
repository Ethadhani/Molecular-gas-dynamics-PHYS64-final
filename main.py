#!/usr/bin/env python3

import numpy as np
from typing import Tuple
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

from scipy.integrate import solve_ivp
import warnings

mpl.rcParams['figure.dpi'] = 150

BOLTZMANN = 0.01381 # k_B Boltzmann constant, units of zeptoJoule/K
IDEALGAS = 8.314e21 # for verification, units zeptoJoule / (K mol)
AVOGADRO = 6.023e23 # Avogadro's number of things per mole
COLLISION_TIME = 1e-9 

# number of points to include in the rolling average of pressure
# (this decreases the number of frames in the pressure plot which introduces a delay)
ROLLING = 10 

# The radius of the sphere is defined as 1 micron
VOLUME = (4/3) * np.pi # units of (micron)^3
SURFACE_AREA = 4 * np.pi # units of (micron)^2

# Conversion factors since we are using microns and nanoseconds
DISTANCE_OVER_TIME_CONVERSION = 1e3
TIME_OVER_DISTANCE_CONVERSION = 1/DISTANCE_OVER_TIME_CONVERSION


# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]

class ParticleSimulator:
    def __init__(self, cuberoot_N: int = 5, temperature = 150, scenario: str = 'ideal', seed = 5):
        '''
            Initializes the particle simulation in a 1-nm radius sphere

            Param:
                cuberoot_N: the cube root of the number of particles in the simulation
                temperature: degrees Kelvin
                scenario: determines the potential function used;
                          either 'ideal', 'attractive', or 'repulsive'
                seed: the number to seed NumPy's RNG with.
        '''
        self.N = cuberoot_N**3
        rng = default_rng(seed=seed)
        # mass of a diatomic nitrogen atom
        self.MASS = 2*2.3259e-5 # units: zepto kg
        # minimal separation between particles for potential calculation
        # needed so that things don't go super fast and break conservation of energy 
        self.MIN_SEPARATION = 1e-20 
        
        if scenario == 'ideal': # potential function which is well-modeled by PV = nRT
            # magic numbers from Elio's desmos         
            self.V0 = 1e5
            self.A = 1e-5 
        elif scenario == 'repulsive': # potential function which causes a lot of repulsion between particles
            self.V0 = 1e4
            self.A = 1e-3
        elif scenario == 'attractive': # potential function that is not ideal but not super repulsive
            # very attractive: it looks good
            self.V0 = 1e5
            self.A = 1e-4
        else:
            raise f"Unknown scenario {scenario}"
    
        self.scenario = scenario

        # generate particles on a grid in a cube of side length 1 inside the sphere
        # this prevents particles from being on top of each other
        self.posX, self.posY, self.posZ = np.meshgrid(
            np.linspace(-0.5, 0.5, cuberoot_N),
            np.linspace(-0.5, 0.5, cuberoot_N),
            np.linspace(-0.5, 0.5, cuberoot_N)
        )

        self.temp = temperature
        self.posX = self.posX.ravel()
        self.posY = self.posY.ravel()
        self.posZ = self.posZ.ravel()

        initial_velocity = np.sqrt(3 * BOLTZMANN * temperature / (self.MASS )) * TIME_OVER_DISTANCE_CONVERSION
        print(f"Initial velocity for T = {temperature} K is {initial_velocity} nm/micros.")
        # generate velocity with constant magnitude v_initial
        # asking numpy RNG to give a number between v_i and v_i each time should yield v_i 
        velocities = self._sphericalToCart(rng.uniform(
            [initial_velocity, 0, 0],
            [initial_velocity, np.pi, 2 * np.pi],  size=(self.N, 3)
        ))
        self.velX = velocities[:, 0]
        self.velY = velocities[:, 1]
        self.velZ = velocities[:, 2]
  
    def checkCollisionsWithSphere(self, pos, vel):
        """Redirects particles which are outside of the unit sphere"""
        norms = np.linalg.norm(pos, axis=1) 

        # indices of all particles outside the unit sphere
        outside_indices = np.where(norms > 1)
 
        # outward normal vector of the sphere which we will use to find the tangent plane
        normal_vector = pos[outside_indices]
        # normalize it
        normal_vector = normal_vector / np.linalg.norm(normal_vector, axis=1, keepdims=True)
    
        # if nothing is outside then return
        if np.size(outside_indices) == 0:
            return pos, vel
        # subtract twice the projection of the velocity onto the tangent plane of the sphere
        # this reflects the change in velocity due to elastic collision with the inside of the sphere
        # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
        # https://stackoverflow.com/questions/68245372/how-to-multiply-each-row-in-matrix-by-its-scalar-in-numpy

        # teleport back into sphere if outside of sphere  
        pos[outside_indices] = normal_vector
        # change the velocity from collision
        
        projected_velocity = (vel[outside_indices] @ np.transpose(normal_vector)).diagonal()[:, None] * normal_vector
        vel[outside_indices] -= 2 * projected_velocity
        magnitude_projected_velocity = np.linalg.norm(projected_velocity, axis=1)

        # force = dP/dt
        # UNITS: zepto-kg * nm / micros (zepto-Newton * s * 10^3)
        total_impulse_exerted = np.sum(2 * self.MASS * magnitude_projected_velocity)
        return pos, vel, total_impulse_exerted
        
    @staticmethod
    def _sphericalToCart(coord):
        '''
            Converts spherical coordinates to Cartesian.
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

        return F
    
    def runIVP(self, t, fpns, hist = False):
        """
            Runs the integration.

            Parameters:
                * t: time in nanoseconds to run for
                * fpns: frames per nanosecond of simulation/second of render
                * hist: whether to show a histogram instead of the ideal-gas-constant plot
        """
        # We can't pass complicated structure through solve_ivp so we make a flat thing
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

        # variable to store data from the different integrations
        dataset = np.zeros((N * 6, fpns * t ))
        frame = 0
        timeList = np.linspace(0, t, int(fpns * t ))
        netImpulseOnSphere = np.zeros(len(timeList))
        propConst = np.zeros(len(timeList)) # stores estimate of ideal gas constants

        tPick = 0 
        while frame < int(fpns * t)-1:
            data = solve_ivp(
                self.dU, t_span=(tPick,t), y0=passedVals,
                t_eval = timeList[frame:],
                max_step = 0.00001, events=event, dense_output=False,
                first_step = COLLISION_TIME,
                # We tried making rtol and atol smaller and it did not help.
            )

            
            # put in place
            frameGen = len(data.t) 
            dataset[:, frame: frame + frameGen] = data.y
            
            # update the frame
            frame += frameGen
            if len(data.y_events[0]) != 0: # if there is a collision (event)
                thisInst = data.y_events[0][0]
                # undo the flattening from before
                pos = np.array([thisInst[:N], thisInst[N:N*2], thisInst[N*2:N*3]]).T
                vel = np.array([thisInst[N*3:N*4], thisInst[4*N:N*5], thisInst[N*5:N*6]]).T

                pos, vel, impulse = self.checkCollisionsWithSphere(pos, vel)
                netImpulseOnSphere[frame] += impulse

                pos = pos.T
                vel = vel.T
                passedVals = np.concatenate((pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
                tPick = data.t_events[0][0]

                # impulse has units of zepto-Newton s * 10^3 
                time_in_micros = (frame/fpns)
                propConst[frame] = (
                    ((np.sum(netImpulseOnSphere[ :frame]) / (time_in_micros))/ SURFACE_AREA) # Pressure
                    * VOLUME / (self.N * self.temp / AVOGADRO)
                ) * DISTANCE_OVER_TIME_CONVERSION**2 

                print(f'frame {frame}, constant/actual constant: {propConst[frame]/IDEALGAS}')

        # the proportionality constant wont be filled in if no collisions during that frame, thus we can fill it in.
        blankSpots = np.where(propConst == 0.0)

        # fill in the blank spots with a loop....
        for i in blankSpots[0][1:]:
            propConst[i] = (
                ((np.sum(netImpulseOnSphere[:i]) / (timeList[i] ) )/SURFACE_AREA) # Pressure
                * VOLUME / (self.N * self.temp / AVOGADRO)
            ) * DISTANCE_OVER_TIME_CONVERSION**2

        # plot the sphere; taken from matplotlib docs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
 
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
 
        print('data has been generated! yay!')
        print(f"Total impulse: {np.sum(netImpulseOnSphere)}, prop constant: {propConst[-3]}")
        print(f'Ideal Gas constant: {IDEALGAS}')

        fig = plt.figure(figsize=plt.figaspect(0.5))

        # plot of either ideal gas constant estimate or histogram of velocities
        ax_prop = None
        prop = None
        if not hist:
            ax_prop = fig.add_subplot(3,2,2)
            prop = ax_prop.plot(timeList[:1], propConst[:1], label='Calc')[0]
            # need to make limits slightly larger than biggest value so it's visible
            ax_prop.set_ylim((np.min(propConst) - 0.5, np.max(propConst) * 1.1))
            ax_prop.set_xlabel('Time (seconds)')
            ax_prop.set_ylabel(r'$\frac{PV}{nT}$')
            ax_prop.set_title('Ideal gas constants')
            ax_prop.axhline(y=IDEALGAS, label='actual', ls=':')
            ax_prop.legend()
        
        # particle plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(x, y, z, alpha=0.1)
       
        xp = dataset[:self.N, :].T
        yp = dataset[self.N:self.N*2, :].T
        zp = dataset[self.N*2:self.N*3, :].T

        xv = dataset[self.N*3:self.N*4, :].T
        yv = dataset[self.N*4:self.N*5, :].T
        zv = dataset[self.N*5:self.N*6, :].T

        KE = (np.square(xv) + np.square(yv) + np.square(zv)) * self.MASS / 2
        KEmin = np.min(KE) * 0.999
        KEmax = np.max(KE) * 1.001 # scale things a little bit to make sure everything's within limits

        print('calculating potential')
        pEtotal = np.zeros(len(timeList))
        for i in range(len(timeList)):
            pEtotal[i]= self.calcPotential(xp,yp,zp,i)

        # plot histogram if needed; we do this here since we need to calculate the energy first
        bins = None
        if hist:
            ax_prop = fig.add_subplot(3, 2, 2)
            ax_prop.set_title('Velocity histogram')
            ax_prop.set_xlabel('Velocity (microns/ns)')
            ax_prop.set_ylabel('count')

            velList = np.sqrt((np.square(xv) + np.square(yv) + np.square(zv)))
            bins = np.linspace(np.min(velList[:-1]), np.max(velList), 20)

            # from matplotlib example
            _, _, prop = ax_prop.hist(velList[0], bins=bins)
        
        scat = ax.scatter(xp[0], yp[0], zp[0], c = KE[0], cmap='cool', 
                          vmin=KEmin, vmax = KEmax)

        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
        ax.set_title(f'Particle simulation (T = {self.temp} K)')
        ax.set_aspect('equal')

        # plot pressure
        presList = moving_average(netImpulseOnSphere, ROLLING) * fpns * DISTANCE_OVER_TIME_CONVERSION/ SURFACE_AREA
        
        ax_pres = fig.add_subplot(3, 2, 4)
        pressure = ax_pres.plot(timeList[:1], presList[:1])[0]
        pred_pressure = TIME_OVER_DISTANCE_CONVERSION * self.N * IDEALGAS * self.temp / (VOLUME * AVOGADRO)
        ax_pres.axhline(y=pred_pressure, c='red', label='Predicted')
        ax_pres.set_ylim((
            min(np.min(presList), pred_pressure * 0.9),
            max(np.max(presList), pred_pressure * 1.1)
        ))

        ax_pres.set_xlabel('Time (sec)')
        ax_pres.set_ylabel(r'Pressure (micro Pascal)')
        ax_pres.set_title(f'Pressure: {np.mean(presList[:1]):.3f} (micro Pascals)')
        ax_pres.legend()

        # energy plots
        kEtotal = np.sum(KE, axis=1)

        # https://matplotlib.org/stable/gallery/spines/multiple_yaxis_with_spines.html
        ax_KE = fig.add_subplot(3,2,6)
        ax_PE = ax_KE.twinx()
        # Plot kinetic and potential energy
        kinetic = ax_KE.plot(timeList[:1], kEtotal[:1], label=r'$K_E$', color = 'steelblue')[0]
        pot = ax_PE.plot(timeList[:1], pEtotal[:1], label = r'$|U_E|$', color='red')[0]

        # Add labels
        ax_KE.set_xlabel('Time (seconds)')
        ax_KE.set_ylabel('$K_E$')
        ax_PE.set_ylabel('$|U_E|$ (J)')

        # set limits and such
        kmin = np.min(kEtotal[5:-5])
        kmax = np.max(kEtotal[5:-5])
        pmin = np.min(pEtotal[5:-5])
        pmax = np.max(pEtotal[5:-5])
        diff = np.abs(np.max([kmax-kmin, pmax-pmin])) * 1.1 / 2

        
        ax_KE.set_title('System energy (zepto Joules)')
        ax_KE.set_ylim((np.median([kmin, kmax]) - diff, np.median([kmin, kmax]) + diff))
        ax_PE.set_ylim((np.median([pmin, pmax]) - diff, np.median([pmin, pmax]) + diff))

        ax_KE.legend(handles = [kinetic, pot])

        # set colors
        ax_KE.tick_params(axis='y', colors=kinetic.get_color())
        ax_PE.tick_params(axis='y', colors=pot.get_color())

        rotateRate = 10 / (fpns)
        def update(frame, prop):
            # particles
            scat._offsets3d = (xp[frame], yp[frame], zp[frame])
            scat.set_array(KE[frame])
            scat.set_clim(vmin=KEmin, vmax=KEmax)

            # Ideal gas constant plot
            if not hist:
                prop.set_xdata(timeList[:frame])
                prop.set_ydata(propConst[:frame])
                ax_prop.set_xlim((0, timeList[frame]+0.1))
            else:
                # from Matplotlib docs "gallery"
                n, _ = np.histogram(velList[frame], bins)
                for count, rect in zip(n, prop.patches):
                    rect.set_height(count)
            if frame % 10: # print some updates
                if not hist:
                    ax_prop.set_title(f'Ideal gas constant: {(propConst[frame]*1e-21):.3f}'+r' J/(mol K)')
                ax_pres.set_title(f'Pressure: {np.mean(presList[:frame]):.3f} (micro Pascals)')
            
            kinetic.set_xdata(timeList[:frame])
            kinetic.set_ydata(kEtotal[:frame])
            pot.set_xdata(timeList[:frame])
            pot.set_ydata(pEtotal[:frame])
            ax_KE.set_xlim((0, timeList[frame] + 0.1))

            pressure.set_xdata(timeList[:frame])
            pressure.set_ydata(presList[:frame] )

            ax_pres.set_xlim((0, timeList[frame] + 0.1))
            # rotates the particle simulation plot in the video
            ax.view_init(30, 60 + rotateRate * frame, 0)

            if not hist:
                return (scat, prop, kinetic, pot, pressure)
            else: # histogram
                return (scat, prop.patches, kinetic, pot, pressure)
        
        fig.tight_layout(pad=.5)

        ani = animation.FuncAnimation(fig = fig, func = update, fargs=(prop,), frames = t * fpns - ROLLING, interval = (1000/fpns) )
        # https://stackoverflow.com/questions/37146420/saving-matplotlib-animation-as-mp4
        ani.save(f'Particles-{self.scenario}-{self.temp}-{self.N}-{t}-{hist}.mp4', writer = animation.FFMpegWriter(fps=fpns))

    def potential(self, a: Coordinate, b: Coordinate) -> Coordinate:
        '''potential function: V(x, y, z) = 
                V_0 (a^4/(x^2 + y^2 + z^2)^2 - a^3/(x^2+y^2+z^2)^(3/2))'''

        radii = a - b
        x = radii[:, 0]
        y = radii[:, 1]
        z = radii[:, 2]
        rsquare = x**2 + y**2 + z**2

        # makes the distance infinity if looking at itself, since particles exert no potential on themselves
        rsquare = np.where(rsquare == 0.0, np.inf, rsquare)
        rsquare_New = np.where(
            rsquare < self.MIN_SEPARATION ** 2, self.MIN_SEPARATION ** 2, rsquare
        )
        with warnings.catch_warnings():
            # suppress divide by 0 warnings
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            potential = self.V0 *  (self.A**4 / rsquare_New**2) - self.V0 * (self.A**3 / rsquare_New**(3/2))

        # get rid of NaN vals from calculating potential with self --- might be redundant?
        # but if it ain't broke..... don't fix it!
        np.nan_to_num(potential, False, nan=0.0)
        return -potential

    def calcPotential(self, xp, yp, zp, frame):
        '''potential function: V(x, y, z) = 
                V_0 (a^4/(x^2 + y^2 + z^2)^2 - a^3/(x^2+y^2+z^2)^(3/2))'''
        
        pos = np.array([xp[frame], yp[frame], zp[frame]]).T
        potentialList = np.apply_along_axis(
            lambda r: np.sum(self.potential(pos, np.tile(r, (self.N,1))), axis = 0),
            axis=1,
            arr=pos
        )
        
        return (np.sum(potentialList)/2)

    def dU(self, t, U):
        '''
            returns the derivative of position and velocity 
            args:
                U unpacks to: x, y, z, vx, vy, vz
            returns:
                vx, vy, vz, ax, ay, az
        '''
        N = self.N
        # unwraps the conditions into a position matrix and a velocity matrix
        pos = np.array([U[:N], U[N:2*N], U[2*N:3*N]]).T
        vel = np.array([U[3*N:4*N], U[4*N:5*N], U[5*N:6*N]])
        
        # makes a matrix for each particle position, calculates all the forces 
        # on it and sums, and divides by mass to get accel
        newAccel = np.apply_along_axis(
            lambda r: np.sum(self.force(pos, np.tile(r, (self.N,1))), axis = 1) / self.MASS,
            axis=1,
            arr=pos
        )

        # re wraps the velocity and acceleration to be passed back
        newAccel = newAccel.T

        return np.concatenate((vel[0], vel[1], vel[2], newAccel[0], newAccel[1], newAccel[2]))

# this function is copied from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

s = ParticleSimulator(scenario='repulsive', cuberoot_N = 5, temperature = 150)
print('Simulation Initiated')
s.runIVP(10, 50, hist=True)
