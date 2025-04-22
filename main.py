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

MASS = 1 # arbitrary unit system
# Potential constants
V0 = 0#0.001
A = 0.1 # initial guess that we can change later
initial_velocity = 1

# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]
#%%

class ParticleSimulator:

    def __init__(self, N: int = 100):
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
    
    def step(self, dt = 0.01) -> None:
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
    
    def checkCollisionsWithSphere(self):
        """Redirects particles which are outside of the unit sphere"""
        norms = np.linalg.norm(self.pos, axis=1)
        # indices of all particles outside the unit sphere
        
        outside_indices = np.where(norms > 1)

        # outward normal vector of the sphere which we will use to find the tangent plane
        normal_vector =  self.pos[outside_indices]
        #normalize it
        normal_vector =  normal_vector / np.linalg.norm(normal_vector, axis=1, keepdims=True)

        # if nothing is outside then return
        if np.size(outside_indices) == 0:
            return

        # subtract twice the projection of the velocity onto the tangent plane of the sphere
        # this reflects the change in velocity due to elastic collision with the inside of the sphere
    #https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    
    #https://stackoverflow.com/questions/68245372/how-to-multiply-each-row-in-matrix-by-its-scalar-in-numpy

        #teleport back into sphere if outside of sphere    
        self.pos[outside_indices] = normal_vector
        #change the velocity from collision

        # print(np.sum(self.vel[outside_indices] @ np.transpose(normal_vector), axis=1)[:, None])
        projected_velocity =(self.vel[outside_indices] @ np.transpose(normal_vector)).diagonal()[:,None] * normal_vector
        self.vel[outside_indices] -= 2 * projected_velocity 
        

    def energy(self) -> None:
        """Kinetic energy"""
        return np.sum(0.5 * MASS * np.square(np.linalg.norm(self.vel, axis = 1)))
    
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
        

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
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

        def update(frame):
            self.step()

            scat._offsets3d = (self.pos[:,0],self.pos[:,1], self.pos[:,2])
            return (scat, )
        
        ani = animation.FuncAnimation(fig = fig, func = update, frames = 1000, interval = 10)
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
        # x,y,z components of r
        # diff = a - b
        # x = diff[0]
        # y = diff[1]
        # z = diff[2]

        x, y, z = a-b

        return np.array([
            -V0 * (A**3 * x * (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2))),
            -V0 * A**3 * y * (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2)),
            -V0 * A**3 * z * (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2))
        ])

s = ParticleSimulator()
#s.run()
s.run()

# %%
