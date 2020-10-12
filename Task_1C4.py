"""
<<<<<<<< Bruker egen kode >>>>>>>>
"""
from ast2000tools.solar_system import SolarSystem
from ast2000tools import utils
from ast2000tools import constants
from ast2000tools import space_mission
import numpy as np
import matplotlib.pyplot as plt
import math as math
""" Finding our special system"""
seed = utils.get_seed("mortetb")
system = SolarSystem(seed)  # 66755

# print(seed)  # 2 siste siffer er 55

"""Constants given by the task, and other info we need"""
lmda0 = 656.28  # nm
c = constants.c
star_name = ["star0_5.93.txt", "star1_1.74.txt", "star2_1.67.txt", "star3_1.72.txt", "star4_0.91.txt"]
vel_list = np.array([0, 10, 18, 14, 34])  # se på indeks nr 3 og spør i gruppetimen
sunmass_list = np.array([5.93, 1.74, 1.67, 1.72, 0.91]) * constants.m_sun  # sunmass in kg
periods = np.array([math.nan, 4500, 5200, 4500, 5000]) * 24 * 3600  # List in days times dayconverter
# Note the periods might be inaccurate, because i kinda had to find them by looking at a graph
print(f"Sunmasses = {sunmass_list}")

"""Functions, in hindsight this should have been a class, buuut then i'd have to 
rewrite the code"""


def read_data(filename):
    # A function made to read a file containing time, lamda and flux
    """
    :param filename: The file you wish to read
    :return: Return the time vecotr in days, the lambda and the flux
    """
    t, lmda, flux = np.loadtxt(filename, usecols=[0, 1, 2], unpack=True)
    return t, lmda, flux


def doppler_velocity(vecLambda):
    # A function which calculates the velocity of the star and the mean velocity
    """
    :param vecLambda: Takes in a vector of lambda values in nm
    :return: The velocity vector of the star in vr [m/2] and the mean velocity of the star
    """
    vtot = c * (vecLambda - lmda0) / lmda0
    meanvel = np.mean(vtot)
    return vtot, meanvel


def leastmass_planet(P, mStar, vStarRadial, G=constants.G):
    # A function which finds the least possible mass of a planet by assuming i = 90 degrees
    """
    :param P: Period
    :param m: Mass of sun in kg
    :param v: radial velocity of the sun
    :param G: Gravitational constant
    :return: least mass of planet
    """
    mp = (((P / (2 * np.pi * G)) ** (1 / 3)) * (mStar ** (2 / 3)) * vStarRadial)
    return mp



def bigHalfAxis_finder(P, m1, m2, G=constants.G):
    # Made to find the big axis of the planetary orbit around a star
    """
    :param P: Period of orbit
    :param m1: mass of sun
    :param m2: mass if planet(this is an estimate)
    :param G: Gravitational constant in kg yadayada
    :return: Radius of big axis
    """
    a = ((P ** 2 * G * (m1 + m2)) / (4 * np.pi ** 2)) ** (1 / 3)
    return a


def vp(a, P):
    # This function finds the velocity of the planet, we assume its constant
    """
    :param a: the big half axis
    :param P: Period of planet orbit
    :return: The velocity of the planet
    """
    vp = (2 * np.pi * a) / P
    return vp


def planetradiusfinder(vStar, vPlanet, t1, t0):
    """
    :param vStar: velocity of star
    :param vPlanet: velocity of planet
    :param t1: time 1, when the planet has fully entered the viw of the sun, input should be in days
    :param t0: time0, when the planet first crosses the view of the sun, input should be in days
    :return: The radius of the Planet
    """
    r = 0.5 * (vStar + vPlanet) * (t1*24*3600 - t0*3600*24)
    return r

def planetDensity(mp,rp):
    rho = mp/(4/3 * np.pi * rp**3)
    return rho


def vr_model(t, P, v, t0):
    # Made to be used in the least square approach as the modelled velocity
    """
    :param t: a time we iterate over
    :param P: The period of the orbital path
    :param v: The radial speed of the star
    :param t0: a constant, the "start" t
    :return:
    """
    vr = v * np.cos(2 * np.pi * (t - t0)/ P)
    return vr



def least_square_method(vr, vStarRadial, P, t0, t):
    delta = np.zeros((len(vStarRadial), len(t0), len(P)))
    for i in range(len(vStarRadial)):
        for j in range(len(t0)):
            for k in range(len(P)):
                delta[i, j, k] = np.sum( ( vr - vr_model(t,P[k],vStarRadial[i],t0[j]) ) **2 )
    minindex, minvalue = np.unravel_index(np.argmin(delta,axis = None),delta.shape), np.min(delta, axis=(0, 1, 2))
    return minindex, minvalue, delta


def plot_flux_vel(filename, save=False):
    """
    This function is made to plot the flux and the velocity of the stars
    :param filename: The filname of the star file you wish to plot
    :param save: If True will save plots, if False will not save plots
    :return: nothing
    """
    time, lmda, flux = read_data(filename)
    vtot, meanvel = doppler_velocity(lmda)
    vr = vtot - meanvel  # i assume we do this to remove the constant speed that the system travels with
    # Plotting the velocity of the star
    plt.subplot(2, 1, 1)
    plt.title(f"Velocity over time for star {filename[4]}")
    plt.xlabel("t [days]")
    plt.ylabel("v [m/s]")
    plt.plot(time, vr)

    # plotting the flux
    plt.subplot(2, 1, 2)
    plt.title(f"Light flux over time for star {filename[4]}")
    plt.ylabel("$F/F_{max}$")
    plt.xlabel("t [days]")
    plt.plot(time, flux)
    if save == True:
        plt.savefig(f"flux_vel_graph_star{filename[4]}")
    plt.show()


"""Plotting the flux and velocity over time"""

for star in star_name:
    plot_flux_vel(star, save=True)

""" Calculating least mass, assuming that i = 90 degrees"""
least_mass_vec = np.zeros(len(vel_list))  # empty list goes brrrrr
for i in range(len(vel_list)):  # Calculates the least mass of the planets in kg
    least_mass_vec[i] = leastmass_planet(periods[i], sunmass_list[i], vel_list[i])
print(f"Least mass = {least_mass_vec}")

"""Getting the values for t, flux and velocity for star nr 2 index 1"""

time, lmda, flux = read_data(star_name[1])
vtot, meanvel = doppler_velocity(lmda)
vpec = vtot - meanvel  # We will need this later, this is the velocity of star 2, index 1, around the center of mass

"""Calculating radius and density"""
t1 = [math.nan,1,1,math.nan,1] # the t1 values for the different systems
a = np.zeros(len(periods)) # An array filled with the radius or "big half axis"
velocity_planets = np.zeros_like(a) # An array filled with velocities
planet_radius = np.zeros_like(a) # This is an array that should be filled with the radius of each planet from core to surface
planet_density = np.zeros_like(a) # An array that shoudl be filled with the density of the planets
for i in range(len(periods)):
    a[i] = bigHalfAxis_finder(periods[i], sunmass_list[i], least_mass_vec[i])

    # We assume circular orbits, this lets us use the fact that the speed would be constant
    velocity_planets[i] = vp(a[i], periods[i])
    planet_radius[i] = planetradiusfinder(vel_list[i],velocity_planets[i],t1[i],0) # We assume that it takes one day, eventhough it does not

    planet_density[i] = planetDensity(least_mass_vec[i],planet_radius[i])

""" Printing the values we calculated over"""
print(f"Planet radius = {planet_radius}")
print(f"Planet density = {planet_density}")
print(f"radius = {a}")
"""Finding the best possible solution for t0, P and vStarRadial"""
# We get this by looking at the plot of star nr 2 and eyeballing it
N = 20
vRadialStar = np.linspace(5, 10, N)
t0 = np.linspace(4000, 5500, N,dtype=int)
P_estimate = np.linspace(4000, 5000, N)

index,minvalue,delta = least_square_method(vpec,vRadialStar,P_estimate,t0,time)
print(index,minvalue)#looking at the index values and the value for delta
print(vRadialStar[index[0]],t0[index[1]],P_estimate[index[2]])

"""We do this functionally cuz reasons"""
plt.subplot(2, 1, 1)
plt.title(f"Velocity over time for star {1}")
plt.xlabel("t [days]")
plt.ylabel("v [m/s]")
plt.plot(time, vpec)
plt.plot(time,vr_model(time,P_estimate[index[2]],vRadialStar[index[0]],t0[index[1]]),"r")
# plotting the flux
plt.subplot(2, 1, 2)
plt.title(f"Light flux over time for star {1}")
plt.ylabel("$F/F_{max}$")
plt.xlabel("t [days]")
plt.plot(time, flux)
plt.savefig("plzwork.jpg")
plt.show()