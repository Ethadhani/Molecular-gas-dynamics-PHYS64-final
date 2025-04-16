#!/usr/bin/env python3
#%%
import numpy as np
from typing import Tuple
from numpy.random import default_rng

MASS = 1 # arbitrary unit system
# Potential constants
V0 = 1 
A = 0.01 # initial guess that we can change later
initial_velocity = 0.05

# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]
#%%

class ParticleSimulator:

    def __init__(self, N: int = 3):
        '''
            Initializes the particle simulation

            Param:
                N: the number of particles in the simulation
        '''
        self.N = N
        rng = default_rng()
        # generate starting positions in spherical coordinates so that we stay withing our bounds
        spherCoord = rng.uniform([0, 0, 0],[1, np.pi, 2 * np.pi], size=(self.N, 3))

        self.pos = [[0,0,0], [0.1,0.2,0.3], [1,1,1]] # self._sphericalToCart(spherCoord)

        # generate velocity with constant magnitude v_initial
        # asking numpy RNG to give a number between v_i and v_i each time should yield v_i 
        self.vel = self._sphericalToCart(rng.uniform(
            [initial_velocity, 0, 0],
            [initial_velocity, np.pi, 2 * np.pi],  size=(self.N, 3)
        ))

        # self.accel = np.zeros((3, self.N))
    
    def step(self, dt = 0.1) -> None:
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
                accel = self.force(self.pos[a_idx], self.pos[b_idx])/MASS
                self.vel[b_idx] += accel * dt
                self.vel[a_idx] -= accel * dt 
        self.pos += self.vel * dt

        # check for wallsl
    
    def energy(self) -> None:
        """Energy"""
        pass

    
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
        print("A is",a)
        x, y, z = a - b
        return (
            -V0 * (a**3 * x (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2))),
            -V0 * a**3 * y (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2)),
            -V0 * a**3 * z (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2))
        )

s = ParticleSimulator()
s.step()
# %%
