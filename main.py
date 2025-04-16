#!/usr/bin/env python3

import numpy as np
from typing import Tuple

MASS = 1 # arbitrary unit system
# Potential constants
V0 = 1 
A = 0.01 # initial guess that we can change later

# type indicating (x, y, z) coordinates
Coordinate = Tuple[float, float, float]

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
    x, y, z = a - b
    return (
        -V0 * (a**3 * x (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2))),
        -V0 * a**3 * y (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2)),
        -V0 * a**3 * z (-(4 * A)/(x**2 + y**2 + z**2)**3 + 3/(x**2 + y**2 + z**2)**(5/2))
    )
