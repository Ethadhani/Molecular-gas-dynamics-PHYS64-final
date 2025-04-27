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
# mpl.rcParams['figure.dpi'] = 200
MASS = 1 # arbitrary unit system
# Potential constants
V0 = 0.1
A = 0.03 # initial guess that we can change later
initial_velocity = 1.0000000
MIN_SEPARATION = 0.003

# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]
#%%

class ParticleSimulator:

    def __init__(self, cuberoot_N: int = 5):
        '''
            Initializes the particle simulation

            Param:
                N: the number of particles in the simulation
        '''
        self.N = cuberoot_N**3
        rng = default_rng()

        # generate starting positions in spherical coordinates so that we stay withing our bounds
        spherCoord = rng.uniform([0, 0, 0], [1, np.pi, 2 * np.pi], size=(self.N, 3))

        # generate particles on a grid in a cube of side length 1.4 inside the sphere
        self.posX, self.posY, self.posZ = np.meshgrid(
            np.linspace(-0.4, 0.4, cuberoot_N),
            np.linspace(-0.4, 0.4, cuberoot_N),
            np.linspace(-0.4, 0.4, cuberoot_N)
        )


        self.posX = self.posX.ravel()
        self.posY = self.posY.ravel()
        self.posZ = self.posZ.ravel()

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
  
    @staticmethod
    def checkCollisionsWithSphere(pos, vel):
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
        #accel[outside_indices] -= 2 * projected_velocity / HOW_LONG_A_COLLISION_TAKES
        #print('vel',projected_velocity)
        return pos, vel
        

    def energy(self) -> float:
        """Kinetic energy"""
        x = np.sum(0.5 * MASS * np.square(np.linalg.norm(self.vel, axis = 1)))
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
        tPick = 0
        while frame < int(fps * t)-1:
            data = solve_ivp(
                self.dU, t_span=(tPick,t), y0=passedVals,
                t_eval = timeList[frame:],
                max_step = 0.001, events=event, dense_output=False,
                first_step = 0.000000001
                #min_step = 0.000001
            )

            
            # times = np.concatenate((times, data.t[1:]))
            # put in place
            frameGen = len(data.t) 
            
            dataset[:, frame: frame + frameGen] = data.y
            
            # update the frame
            frame +=frameGen
            if len(data.y_events[0]) != 0:

                thisInst = data.y_events[0][0]
                pos = np.array([thisInst[:N], thisInst[N:N*2], thisInst[N*2:N*3]]).T
                vel = np.array([thisInst[N*3:N*4], thisInst[4*N:N*5], thisInst[N*5:N*6]]).T
                pos, vel = self.checkCollisionsWithSphere(pos, vel)
                pos = pos.T
                vel = vel.T
                passedVals = np.concatenate((pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
                tPick = data.t_events[0][0]
                print('frame ',frame)
        #plot the sphere taken from matplotlib docs
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
 
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
 
        print('data has been generated! yay!')

        fig = plt.figure()#figsize#=plt.figaspect(2.))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, z, alpha=0.1)

        # ax_2d = fig.add_subplot(2,1,2)

        
        # vmin = 0
        # vmax = 2 * energies[0] / self.N
        # fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap='hot_r'),
        #      ax=ax, orientation='vertical', label='Kinetic Energy')

        #plot sphere
        # ax.scatter(sphere[:,0], sphere[:,1], sphere[:,2], alpha =0.2)


        # get the positions in plottable form
        # data.y is NOT the y coordinate, just the solve_ivp output
        posData = data.y
        # print(np.shape(data.y))
        # xp = posData[:self.N, 0]
        # yp = posData[:self.N, 1]
        # zp = posData[:self.N, 3]
        xp = dataset[:self.N,:].T
        yp = dataset[self.N:self.N*2,:].T
        zp = dataset[self.N*2:self.N*3,:].T

        
        scat = ax.scatter(xp[0], yp[0], zp[0])# c=np.linalg.norm(self.vel, axis=1), cmap="cool",
                #    )#vmax=vmax, vmin=vmin)
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))
        ax.set_aspect('equal')

        def update(frame):
            scat._offsets3d = (xp[frame], yp[frame], zp[frame])
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
        
        ani = animation.FuncAnimation(fig = fig, func = update, frames = t * fps, interval = (1000/fps) - 1)
        # https://stackoverflow.com/questions/37146420/saving-matplotlib-animation-as-mp4
        #ani.save('Particles.mp4', writer = animation.FFMpegWriter(fps=fps))
        plt.show()


    
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
            lambda r: np.sum(self.force(pos, np.tile(r, (self.N,1))), axis = 1) / MASS,
            axis=1,
            arr=pos
        )

        # re wraps the velocity and acceloration to be passed back
        newAccel = newAccel.T

        return np.concatenate((vel[0], vel[1], vel[2], newAccel[0], newAccel[1], newAccel[2]))

s = ParticleSimulator()
print('Simulatio Initiated')
#s.run()
# s.runPre(0.01, 5)
s.runIVP(4, 40)
# %%
