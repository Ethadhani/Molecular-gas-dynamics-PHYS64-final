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
mpl.rcParams['figure.dpi'] = 200
MASS = 1 # arbitrary unit system
# Potential constants
V0 = 0.01
A = 0.02 # initial guess that we can change later
MIN_SEPARATION = 0.005
initial_velocity = 1

# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]
#%%

class ParticleSimulator:

    def __init__(self, N: int = 50):
        '''
            Initializes the particle simulation

            Param:
                N: the number of particles in the simulation
        '''
        self.N = N
        rng = default_rng()
        # generate starting positions in spherical coordinates so that we stay withing our bounds
        spherCoord = rng.uniform([0, 0, 0],[1, np.pi, 2 * np.pi], size=(self.N, 3))

        self.pos = self._sphericalToCart(spherCoord)

        # generate velocity with constant magnitude v_initial
        # asking numpy RNG to give a number between v_i and v_i each time should yield v_i 
        self.vel = self._sphericalToCart(rng.uniform(
            [initial_velocity, 0, 0],
            [initial_velocity, np.pi, 2 * np.pi],  size=(self.N, 3)
        ))

        # self.accel = np.zeros((3, self.N))
    
    def step(self, dt) -> None:
        """
            Runs one time step of the simulation
        """
        # for a in self.pos:
        #     for b in self.pos
        #         stuff(a,b)
        # see https://stackoverflow.com/questions/60559050/python-apply-function-over-pairwise-elements-of-array-without-using-loops
        # https://numpy.org/doc/2.2/reference/generated/numpy.meshgrid.html
        # print(np.dot(self.pos, self.pos))
        # # https://stackoverflow.com/questions/57032234/multidimensional-numpy-outer-without-flatten
        # all_pairwise_particle_distances = np.multiply.outer(self.pos ,  np.ones((self.N)))
        # print("hi", (all_pairwise_particle_distances - all_pairwise_particle_distances.T).T)
        # forces = self.force(*np.meshgrid(self.pos, self.pos, sparse=True))
        # print(forces, self.pos)
        for a_idx in range(self.N):
            for b_idx in range(a_idx + 1, self.N): # +1 so particles dont explode
                accel = self.force(self.pos[a_idx], self.pos[b_idx]) #/ MASS
                self.vel[b_idx] += accel * dt
                self.vel[a_idx] -= accel * dt 
        self.pos += self.vel * dt

        # check if we have exited the wall,
        
        #indicies = np.where(norms >=1)
        # np.where maybe to swap velocities, 
        # take the tangent plane of sphere, swap velocity that is normal to it which should be same as subtracting twice projection
        # https://math.stackexchange.com/questions/633181/formula-to-project-a-vector-onto-a-plane

        # check for walls with tangent pl

        self.checkCollisionsWithSphere()
    
    @staticmethod
    def checkCollisionsWithSphere(pos, vel):
        """Redirects particles which are outside of the unit sphere"""
        norms = np.linalg.norm(pos, axis=1)
        # indices of all particles outside the unit sphere
        
        outside_indices = np.where(norms > 1)

        # outward normal vector of the sphere which we will use to find the tangent plane
        normal_vector = pos[outside_indices]
        #normalize it
        normal_vector = normal_vector / np.linalg.norm(normal_vector, axis=1, keepdims=True)

        # if nothing is outside then return
        if np.size(outside_indices) == 0:
            return vel

        # subtract twice the projection of the velocity onto the tangent plane of the sphere
        # this reflects the change in velocity due to elastic collision with the inside of the sphere
    #https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    
    #https://stackoverflow.com/questions/68245372/how-to-multiply-each-row-in-matrix-by-its-scalar-in-numpy

        #teleport back into sphere if outside of sphere    
        pos[outside_indices] = normal_vector
        #change the velocity from collision

        # print(np.sum(self.vel[outside_indices] @ np.transpose(normal_vector), axis=1)[:, None])
        projected_velocity = (vel[outside_indices] @ np.transpose(normal_vector)).diagonal()[:,None] * normal_vector
        vel[outside_indices] -= 2 * projected_velocity 
        return vel
        

    def energy(self) -> float:
        """Kinetic energy"""
        x = np.sum(0.5 * MASS * np.square(np.linalg.norm(self.vel, axis = 1)))
        # print(x)
        return x
    
    def run(self):
        '''

        '''
        # taken from matplotlib documentation and research code

        #plot the sphere taken from matplotlib docs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        

        fig = plt.figure()#figsize=plt.figaspect(2.))
        ax = fig.add_subplot(projection='3d')
        # ax_2d = fig.add_subplot(2,1,2)

        energies = [self.energy()]
        times = [0]

        # energy_plot = ax_2d.plot(times, energies)[0]
        # ax.plot_surface(x, y, z, alpha=0.1)
        vmin = 0
        vmax = 2 * self.energy() / self.N
        # fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='hot_r'),
        #      ax=ax, orientation='vertical', label='Kinetic Energy')

        # #plot sphere
        # ax.scatter(sphere[:,0], sphere[:,1], sphere[:,2], alpha =0.2)


        # get the positions in plottable form
        xp = self.pos[:,0]
        yp = self.pos[:,1]
        zp = self.pos[:,2]
        
        scat = ax.scatter(xp, yp, zp, c=0.5 * np.linalg.norm(self.vel, axis=1)**2, cmap="hot_r",
                    vmax=vmax, vmin=vmin)
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))

        
        # define update function
        dt = 0.002
        # ax_2d.set_xscale('log')
        # ax_2d.set_yscale('log')

        def update(frame):
            n = 5
            for _ in range(n):
                self.step(dt)
            # energies.append(self.energy())
            # times.append(times[-1] + dt * n)

            scat._offsets3d = (self.pos[:,0],self.pos[:,1], self.pos[:,2])
            scat.set_array(0.5 * np.linalg.norm(self.vel, axis=1)**2)
            # energy_plot.set_xdata(times)
            # energy_plot.set_ydata(energies)
            # ax_2d.set_xlim([0, times[-1]])
            # ax_2d.set_ylim([0, max(energies) * 2])
            return (scat, )
        
        ani = animation.FuncAnimation(fig = fig, func = update, frames = 1000, interval = 3)
        plt.show()

    def runPre(self, dt, time):


        # pre compute

        numdt = int(time / dt)

        posData = np.zeros((numdt, self.N, 3))
        velData = np.zeros((numdt, self.N, 3))
        KData = np.zeros(numdt)
        timeData = np.zeros(numdt)

        time = 0
        for i in range(numdt):
            posData[i] = self.pos
            velData[i] = self.vel
            KData[i] = self.energy()
            if KData[i] == 0:
                KData[i] = 0.01
            timeData[i] = time
            if i % 10 == 0:
                print(time)
            time+= dt
            self.step(dt)
        KData = np.abs(KData)

        print('data has been generated! yay!')

        fig = plt.figure(figsize=plt.figaspect(2.))
        ax = fig.add_subplot(2,1,1,projection='3d')
        ax_2d = fig.add_subplot(2,1,2)

        times = []
        energies = []
        energy_plot = ax_2d.plot(times, energies)[0]
        # ax.plot_surface(x, y, z, alpha=0.1)
        # vmin = 0
        # vmax = 2 * energies[0] / self.N
        # fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='hot_r'),
        #      ax=ax, orientation='vertical', label='Kinetic Energy')

        # #plot sphere
        # ax.scatter(sphere[:,0], sphere[:,1], sphere[:,2], alpha =0.2)


        # get the positions in plottable form
        xp = posData[0][:,0]
        yp = posData[0][:,1]
        zp = posData[0][:,2]
        
        scat = ax.scatter(xp, yp, zp, c=np.linalg.norm(self.vel, axis=1), cmap="cool",
                    )#vmax=vmax, vmin=vmin)
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))

        def update(frame):
            scat._offsets3d = (posData[frame][:,0], posData[frame][:,1], posData[frame][:,2])
            scat.set_array(np.linalg.norm(velData[frame], axis=1))

            energy_plot.set_xdata(timeData[:frame])
            energy_plot.set_ydata(KData[:frame])
            ax_2d.set_xlim((0, timeData[frame]))
            ax_2d.set_ylim((0, KData[frame])) # assumption: energy never decreases
            # if frame == 3: # FIXME: this so it is not zero which makes mat plot lib yell at us
            #     print(KData[2:frame])
            #     ax_2d.set_yscale('log')
                # sys.exit()
            #     ax_2d.set_xscale('log')
            return (scat, )
        
        ani = animation.FuncAnimation(fig = fig, func = update, frames = numdt, interval = dt * 1000)
        plt.show()


    
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
    
    @staticmethod
    def force(a: Coordinate, b: Coordinate) -> Coordinate:
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
        rsquare_corrected = np.where(rsquare_real < MIN_SEPARATION ** 2, MIN_SEPARATION ** 2, rsquare_real)

        F = np.array([
            -V0 * (A**3 * x * (-(4 * A)/(rsquare_corrected)**3 + 3/(rsquare_corrected)**(5/2))),
            -V0 * A**3 * y * (-(4 * A)/(rsquare_corrected)**3 + 3/(rsquare_corrected)**(5/2)),
            -V0 * A**3 * z * (-(4 * A)/(rsquare_corrected)**3 + 3/(rsquare_corrected)**(5/2))
        ])
        F[:,zero_radius_idx] = 0 # remove force of a particle on itself
        # print(F)
        # An array has 
        # n = np.linalg.norm(F)
        # if n > 1:
        #     print(n)
        return F
    
    def runIVP(self, t, fps):
        passedVals = np.zeros((self.N * 2, 3))
        # print(passedVals)
        passedVals[: self.N, :] = self.pos
        passedVals[self.N : , :] = self.vel
        data = solve_ivp(self.dU, t_span=(0,t), y0=passedVals.ravel(), t_eval= np.linspace(0, t, fps * t))
        
        print('data has been generated! yay!')

        fig = plt.figure()#figsize#=plt.figaspect(2.))
        ax = fig.add_subplot(projection='3d')
        # ax_2d = fig.add_subplot(2,1,2)

        times = data.t        # ax.plot_surface(x, y, z, alpha=0.1)
        # vmin = 0
        # vmax = 2 * energies[0] / self.N
        # fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='hot_r'),
        #      ax=ax, orientation='vertical', label='Kinetic Energy')

        #plot sphere
        ax.scatter(sphere[:,0], sphere[:,1], sphere[:,2], alpha =0.2)


        # get the positions in plottable form
        # data.y is NOT the y coordinate, just the solve_ivp output
        posData = data.y
        # print(np.shape(data.y))
        # xp = posData[:self.N, 0]
        # yp = posData[:self.N, 1]
        # zp = posData[:self.N, 3]
        xp = posData[:self.N,:]
        yp = posData[self.N:self.N*2,:]
        zp = posData[self.N*2:self.N*3,:]

        
        scat = ax.scatter(xp[:,0], yp[:,0], zp[:,0])# c=np.linalg.norm(self.vel, axis=1), cmap="cool",
                #    )#vmax=vmax, vmin=vmin)
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))

        def update(frame):
            scat._offsets3d = (xp[:,frame], yp[:,frame], zp[:,frame])
            # scat.set_array(np.linalg.norm(velData[frame], axis=1))

            # energy_plot.set_xdata(timeData[:frame])
            # energy_plot.set_ydata(KData[:frame])
            # ax_2d.set_xlim((0, times[frame]))
            # ax_2d.set_ylim((0, KData[frame])) # assumption: energy never decreases
            # if frame == 3: # FIXME: this so it is not zero which makes mat plot lib yell at us
            #     print(KData[2:frame])
            #     ax_2d.set_yscale('log')
                # sys.exit()
            #     ax_2d.set_xscale('log')
            return (scat, )
        
        ani = animation.FuncAnimation(fig = fig, func = update, frames = t * fps, interval = 1000/fps)
        # https://stackoverflow.com/questions/37146420/saving-matplotlib-animation-as-mp4
        ani.save('Particles.mp4', writer = animation.FFMpegWriter(fps=fps))
        # plt.show()


    
    def dU(self, t, U):
        '''
            returns the derivative of position and velocity 
            args:
                U unpacks to: x, y, z, vx, vy, vz
            returns:
                vx, vy, vz, ax, ay, az
        '''

        # unwraps the conditions into a position matrix and a velocity matrix.
        pos, vel = U.reshape(2, self.N, 3)
        
        # print("vel", vel)
        newVel = self.checkCollisionsWithSphere(pos, vel)

        # makes a matrix for each particle position, calculates all the forces 
        # on it and sums, and divides by mass to get accel
        newAccel = np.apply_along_axis(
            lambda r: np.sum(self.force(pos, np.tile(r, (self.N,1))), axis = 1) / MASS,
            axis=1,
            arr=pos
        )

        # re wraps the velocity and acceloration to be passed back
        passedVals = np.zeros((self.N * 2, 3))
        passedVals[: self.N, :] = newVel
        passedVals[self.N : , :] = newAccel
        
        return passedVals.ravel()

s = ParticleSimulator()
#s.run()
# s.runPre(0.01, 5)
s.runIVP(10, 45)
# %%
