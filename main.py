# -*- coding: utf-8 -*-

import collections

import math
import numpy as np
from matplotlib import pyplot

def _to_it(x):
    """If x is not iterable, puts it inside one"""
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)

def integrate(f, T, y0, t0=0, N=00000, params=None, check_photon_sphere=True):
    """RK4 integrator
    Solves y' = f(t, y, *params) over interval T
    y may be a vector
    N: number of points to calculate; returns N+1 points
    """
    y0 = np.array(y0)
    if params is None:
        params = ()
        def g(t, y, *p):
            return f(t, y)
    else:
        g = f
    y = np.zeros([N+1,len(_to_it(y0))])
    y[0] = y0
    last_point = 0
    for n in range(N):
        if check_photon_sphere:
            if y[n][0] < 1.5:
                break
        h = (T/N) * (1 - 0.95/y[n][0])
        k1 = g(t0 + n*h, y[n], *params)
        k2 = g(t0 + n*h + h/2, y[n] + (h/2)*k1, *params)
        k3 = g(t0 + n*h + h/2, y[n] + (h/2)*k2, *params)
        k4 = g(t0 + n*h + h, y[n] + h*k3, *params)
        y[n+1] = y[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        last_point += 1
    return (y, last_point)

n_points = 10000

# Measured in units of Sch. radius
r0 = 50

def F(t, y, b):
    """y is a tuple (r,v,phi)"""
    (r, v, phi) = y
    acceleration = 0.5*(-1+r)*(2*r**3 + b**2 * (5-7*r+2*r**2))/r**6
    return np.array((v, acceleration, b*(1-1/r)/r**2))

n_rays = 200

pyplot.figure()
pyplot.xlim(-r0/2, r0/2)
pyplot.ylim(-r0/2, r0/2)
pyplot.axes().set_aspect("equal")

pyplot.plot([0],[0],"ok")

#Draw horizon and photon sphere
r_h = 1
r_ps = 1.5
phi = np.linspace(0, 2*math.pi, 1000)
pyplot.plot(r_h*np.cos(phi), r_h*np.sin(phi), 'k')
pyplot.plot(r_ps*np.cos(phi), r_ps*np.sin(phi), 'k')

ps_angle = math.atan(r_ps/r0) # half angle of the photon sphere
#actually photons will fall in even if the starting angle is greater, because
#they get deflected inwards (duh)

b_ps = 9/2
beta_ps = -math.acos(math.sqrt(1-1/r0)*b_ps/r0)
alpha_ps = math.pi/2 - beta_ps

def _cubic(x):
    """Maps [0,1] to [0,1] like a cubic: flatter near x=1/2"""
    return 0.5 * (2*(x-1/2))**3 + 1/2

for i in range(n_rays):
    # alpha is measured from x-axis up, beta is measured from r=const to the right
    alpha = 3*math.pi/4 + ((i/n_rays-1)**9 + 1) * math.pi/4
    beta = math.pi/2 - alpha
    b = math.cos(beta) * r0 / math.sqrt(1-1/r0)
    v0 = math.sin(beta) * (1-1/r0)
    (data, last_point) = integrate(F, 200, (r0,v0,0), N=n_points, params=(b,))
    (r, v, phi) = np.transpose(data[0:last_point+1])
    (x, y) = (r*np.cos(phi), r*np.sin(phi))
    pyplot.plot(x, y)
    pyplot.pause(0.0001)
    r_min = np.min(r)
    print(i, alpha, last_point, b, r_min)
    if r_min < 1.5:
        break

pyplot.show()
