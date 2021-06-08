#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:35:03 2021

@author: indujaa
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:34:38 2020

Stokes matrix for different polarizations:
H - [1 1 0 0]
V - [1 -1 0 0]
LC - [2 0 0 -2]
RC - [2 0 0 2]
+45 linear - [1 0 1 0]
-45 linear - [1 0 -1 0]  

Things to add:
    
    1. Make seperate functions for noncpherent transmission from Fung (use only I2EM for now) aka make crosspol terms nonzero for transmittivity

    3. VRT for Magellan emissivity
    4. Correct way of calculating CPR

    6. EDEinv is messed up
    
Radar CS cs backscatter coefficient: 
Bruce's white paper says the RCS is converted to BSC during SAR processing; then scaled to Muhleman value.
So the Magellan BSC should be comparable to sigma0

Multiple scattering can be ignored for volfrac < 0.1 in a lossy medium (Fa 2011, Tsang 1985, Jun 1984)

Small volfrac combined with large rock size would produce the same effect as large volfrac of small rocks

@author: Indujaa
"""
import sys
import numpy as np
from cmath import *
from decimal import *
from scipy.constants import *
import scipy.integrate
from scipy.special import gamma

import matlab.engine

from pytmatrix import tmatrix, orientation, scatter
from pytmatrix.psd import PSDIntegrator, GammaPSD, ExponentialPSD

import radar_inst as radar
#from Layers import Layers
from RoughSurface import RoughSurface
import FresnelCoefficients as Fresnel

# np.set_printoptions(formatter={'complex_kind': '{:.8f}'.format},suppress = True)
# getcontext().prec = 10

class VRT:
    """
    Vector Radiative Transfer (VRT) class.
    
    A class containing all the attributes and methods for the VRT model. 
    Each VRT run requires an instance of the class VRT.
    
    Attributes:
    layers: 3 instances of the class Layerrs; one each for the atmosphere,
            upper ground layer, and the substrate layer
            
    Returns:
    VRT model object initialized with the input physical layer startigraphy.
    """
    
    
    def __init__(self, layers):

        # Constants
        self.wavelength = radar.wavelength                     # wavelength in meters
        self.nu = c / self.wavelength
        self.k = 2 * np.pi / self.wavelength
        
         ## Magellan altitude at periapsis
        self.R = radar.altitude                   # 250 km periapsis; 8029 km apoapsis
        
        # Polarization information
        self.polV = radar.polarization1
        self.polH = radar.polarization2
        
        ## Magellan incidence angles ranged from 16 - 45 deg
        ## Incident and scattered azimuth angles are the same for monostatic radar
       
        self.theta_i = np.deg2rad(radar.incidence_angle)      
       
        # Layers
        self.atm = layers[0]
        self.l1= layers[1]
        self.l2 = layers[2]
       
     
        # Final backscatter
        self.sigma = {}
 
    
    def setLayerProperties(self):
        
        """
        Calculate any physical and EM properties of any of the layers from the inputs.
        Properties calculated: (1) Refractive index.
        """
       
        for layer in [self.atm, self.l1, self.l2]:
            layer.ri = layer.eps ** 0.5
            try:
                layer.inclusions.ri = (layer.inclusions.eps/layer.eps) ** 0.5
                
            except:
                pass
                       
    
    def setGeometricOptics(self):
        
        """
        Sets the viewing geometry properties (mainly angles) for each layer.
        """
        
        # # set theta and phi for backscatter
        self.theta_s = self.theta_i
        #theta_s = np.deg2rad(25)
        self.phi_i = np.deg2rad(0)
        self.phi_s = np.pi - self.phi_i
        
        self.l1.k = self.k * np.sqrt(self.l1.eps.real) * np.cos(self.theta_i)
        
        # # Incident, scattering and transmission angles at the top boundary of layer l_i
        
        self.l1.theta_i = self.theta_i
        self.l1.theta_s = self.theta_i
        self.l1.theta_t = self.TransmissionAngle(self.atm, self.l1)
        self.l2.theta_i = self.l1.theta_t
        self.l2.theta_s = self.l2.theta_i
        self.l2.theta_t = self.TransmissionAngle(self.l1, self.l2)
        
    
    def TransmissionAngle(self, l1, l2):
        """
        Compute the angle of transmisssion (rad) from one medium to another.
        
        Args:
        l1: upper medium (instance of class RoughSurface)
        l2: lower medium (instance of class RoughSurface)
        
        Returns:
        Angle of transmission (in radians) from l1 to l2
        """
        mu1 = np.cos(l2.theta_i)
        n = np.sqrt(l2.eps/l1.eps)
        
        mu2 = np.sqrt(1.0 - ((1.0 - mu1**2) / n**2)).real
        theta_t = np.arccos(mu2)
#        theta_t = np.arcsin(l1.ri.real * np.sin(l2.theta_i) / l2.ri.real)
        return theta_t

       
    def FresnelCoefH(self, l1, l2, theta_i):
        
        """ 
        Compute the Fresnel Reflection and Transmission coefficients, and
        Fresnel Reflectivity and Transmissivity for horizontally polarized wave.
        formula from Ulaby and Long, 2014 - page 61 (or) 476.
        
        Args:
        l1: upper medium (instance of class RoughSurface)
        l2: lower medium (instance of class RoughSurface)
        theta_i: Incidence angle (in radians) 
        
        Returns:
        An array of FRC, FTC, reflectivity, and transmissivity for H polarized wave
        
        """
        mu1 = np.cos(theta_i)
        n = np.sqrt(l2.eps/l1.eps)
        
        mu2 = np.sqrt(1.0 - (1.0 - mu1**2) / n**2).real
        
        refh = (mu1 - n * mu2) / (mu1 + n * mu2)

        transh = 2*mu1 / (mu1 + n*mu2)
#         transh = 1 + refh
        rh = refh.real**2 + refh.imag**2
#         th - 1 - rh
        th = (n*mu2/mu1).real * np.abs(transh) ** 2

        return np.array([refh, transh, rh, th])
       
    
    def FresnelCoefV(self, l1, l2, theta_i):
        
        """ 
        Compute the Fresnel Reflection and Transmission coefficients, and
        Fresnel Reflectivity and Transmissivity for vertically polarized wave.
        formula from Ulaby and Long, 2014 - page 61 (or) 476.
        
        Args:
        l1: upper medium (instance of class Layers)
        l2: lower medium (instance of class Layers)
        theta_i: Incidence angle (in radians) 
        
        Returns:
        An array of FRC, FTC, reflectivity, and transmissivity for V polarized wave
        
        """
        mu1 = np.cos(theta_i)
        n = np.sqrt(l2.eps/l1.eps)
        
        mu2 = np.sqrt(1.0 - (1.0 - mu1**2) / n**2).real
        
        refv = (mu2 - n * mu1) / (mu2 + n * mu1)

        transv = 2*mu1 / (mu2 + n*mu1)
#         transv = (1 + refv) * mu2 / mu1
        rv = refv.real**2 + refv.imag**2
#         tv = 1 - rv
        tv = (n*mu2/mu1).real * np.abs(transv) ** 2

        return np.array([refv, transv, rv, tv])

    
    def CoherentTransMatrix(self, l1, l2, theta_i):
        
        """ 
        Compute the 4x4 real-valued transmission matrix 
        
        # # adapted from Fung 1994 + SMRT's iem_fung92.py - SMRT has an additional mu2/mu1 in tvh
        # # value of tvh calculated from smrt's fresnel.py
        # #  add T_coh *= transmittivity in v and h
        
        Args:
        l1: upper medium (instance of class Layers)
        l2: lower medium (instance of class Layers)
        theta_i: Incidence angle (in radians) 
        
        Returns:
        A 4x4 numpy array representing the coherent transmission matrix
        
        """
        
        mu1 = np.cos(theta_i)
        n = np.sqrt(l2.eps/l1.eps)

        
        mu2 = np.sqrt(1.0 - (1.0 - mu1**2) / n**2).real
        
        # # currently hardcoded to always use rms height of layer 1 - change later
        upperks = self.l1.upperks
        
        refh,transh,rh,th = self.FresnelCoefH(l1, l2, theta_i)
        refv,transv,rv,tv = self.FresnelCoefV(l1, l2, theta_i)

#         # # adapted from line 78, func fresnel_transmission matrix 
#         # # in SMRT's core/fresnel.py
#         tvh2 = (n*mu2/mu1).real * (transv * np.conj(transh)).real                      
#         thv2 = (n*mu2/mu1).real * (transh * np.conj(transv)).real 
        
#         tvh = np.sqrt(transv * np.conj(transh))
#         thv = np.sqrt(transh * np.conj(transv))
# #         thv = -tvh

        mu_i = np.cos(theta_i)
        
        K_i = self.k * np.sqrt(l1.eps.real) * mu_i
        K_t = self.k * np.sqrt(l2.eps - ((1 - mu_i**2) * l1.eps)).real
        loss_factor = np.exp(-upperks**2 * (K_t - K_i)**2)
        
        # # adding a factor in front of transmission matrix based on equation 11.61b in Ulaby big book
        fact = (n**3 * mu2 / mu1).real
        
        r11 = np.abs(transv) ** 2
        r22 = np.abs(transh) ** 2
        r33 = (transv * np.conj(transh)).real
        r43 = (transv * np.conj(transh)).imag
        T = fact * np.matrix([[r11, 0., 0., 0.], [0., r22, 0., 0.], [0., 0., r33, -r43], [0., 0., r43, r33]])
        T_coh = loss_factor * T
        
        return T_coh
    
    def CoherentRefMatrix(self, l1, l2, theta_i):
        
        """ 
        Compute the 4x4 real-valued reflection matrix 
        
        # # adapted from Fung 1994 + SMRT's iem_fung92.py - SMRT has an additional mu2/mu1 in tvh
        # # value of rvh (r33, r43 and the like) calculated from smrt's fresnel.py
        # #  add R_coh *= reflectivity in v and h
        
        Args:
        l1: upper medium (instance of class Layers)
        l2: lower medium (instance of class Layers)
        theta_i: Incidence angle (in radians) 
        
        Returns:
        A 4x4 numpy array representing the coherent reflection matrix
        
        """
        
        refh,_,rh,_ = self.FresnelCoefH(l1, l2, theta_i)
        refv,_,rv,_ = self.FresnelCoefV(l1, l2, theta_i)

        mu = np.cos(theta_i)                                                      # for specular reflection, mu_i = mu_s = mu
        K_s2 = self.k ** 2 * (l1.eps.real ** 2 + l1.eps.imag ** 2)                # square of the wavenumber in the scattered medium
        
        loss_factor = np.exp(-2 * l2.upperks**2 * K_s2 * mu**2)
        
        r11 = rh
        r22 = rv
        r33 = (refv * np.conj(refh)).real
        r43 = (refv * np.conj(refh)).imag
        R = np.matrix([[r11, 0., 0., 0.], [0., r22, 0., 0.], [0., 0., r33, -r43], [0., 0., r43, r33]])
        R_coh = loss_factor * R

        return R_coh
        
    def NonCoherentRefMatrix(self, interface, theta_i):
        
        """ 
        Compute the 4x4 real-valued matrix containing backscatter coefficients from a rough interface using I2EM.
        Uses formula from Fung 2002 paper.
        
        # # Uses I2EM surface backscatter model written in matlab using matlab engine for python. 
        
        Args:
        interface: interface between upper and lower medium (instance of class RoughSurface)
        theta_i: Incidence angle (in radians) 
        
        Returns:
        A 4x4 numpy array representing the backscatter matrix containing I2EM sigma values.
        
        """

#         theta_i = np.rad2deg(theta1)
        
#         eng = matlab.engine.start_matlab()
#         sigma0vv, sigma0hh, sigma0hv = eng.I2EM_Backscatter_model(self.nu/1e9, l2.upperks, l2.corrlen, theta_i.item(), l2.eps/l1.eps, 1, 0, nargout=3)           # using .item() to convert from numpy float64 to python scalar
#         svv, shh, svh = np.sqrt(self.sigma02sigma([sigma0vv, sigma0hh, sigma0hv]))
    
        refh,_,rh,_ = Fresnel.FresnelH(interface.eps1, interface.eps2, interface.theta_i)
        refv,_,rv,_ = Fresnel.FresnelV(interface.eps1, interface.eps2, interface.theta_i)
        
        refv = -refv

        mu = np.cos(interface.theta_i)                                                            # for specular reflection, mu_i = mu_s = mu
        K_s2 = self.k ** 2 * (interface.eps1.real ** 2 + interface.eps1.imag ** 2)                # square of the wavenumber in the scattered medium
        
        loss_factor = np.exp(-2 * interface.sigma*1e-2**2 * K_s2 * mu**2)
        
        r11 = rh
        r22 = rv
        r33 = (refv * np.conj(refh)).real * loss_factor * (4 * np.pi * np.cos(theta_i))
        r43 = (refv * np.conj(refh)).imag * loss_factor * (4 * np.pi * np.cos(theta_i))
                
        svv, shh = interface.Copol_BSC()

        R = np.matrix([[svv, 0., 0., 0.], [0., shh, 0., 0.], [0., 0., r33, -r43], [0., 0., r43, r33]]) / (4 * np.pi * np.cos(theta_i))
        return R
        
    def NonCoherentTransMatrix(self, interface, theta_i):
        
        """ 
        Compute the 4x4 real-valued matrix containing transmitted scatter coefficients from a rough interface.
        Uses formula from Fung 1992, Appendix 4D.
        
        # # CURRENTLY DOESN'T WORK. NEEDS FIXING. 
        
        Args:
        interface: interface between upper and lower medium (instance of class RoughSurface)
        theta_i: Incidence angle (in radians) 
        
        Returns:
        A 4x4 numpy array representing the transmitted scattering coeffecient matrix
        
        """
        stvv, sthh = interface.Copol_BSC_trans()

        T = np.matrix([[stvv, 0., 0., 0.], [0., sthh, 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]) / (4 * np.pi * np.cos(theta_i))
        return T
              

    def numscatterers_pervolume(self, psd, dist_type="Gamma", nf=1, Lambda=1, mu=.01, D_max = 0.06, D_med = 1):
        
        """ 
        Compute the integrated total number concentration of particles in the upper layer.
        
        # # CURRENTLY DOESN'T WORK. NEEDS FIXING. 
        
        Args:
        psd: particla size distribution (instance of pytmatrix.psd class)
        dist_type: Type of distribution. Curren options:"Exponential" or Gamma"
        nf: nf or Nw paramter use din the pytmatrix.psd class 
        Lambda: shape paramter for exponential distribution
        mu: paramter for gamma disstribution
        D_max: maximum particle size/diamter
        D_med: median particle size/diamter
        
        Returns:
        Total number concentration of particles per unit volume (?)
        
        """
        
        if dist_type == "Gamma":
            nf = nf * 6.0/3.67**4 * (3.67+mu)**(mu+4)/gamma(mu+4)
            Dfunc = lambda d: nf * np.exp(mu*np.log(d)-(3.67+mu)*(d))
            Dmax = D_max/D_med
            
        elif dist_type == "Exponential":
            Dfunc = lambda d: nf * np.exp(-Lambda*d)
            Dmax = D_max
        
        D_func_int, err = scipy.integrate.quad(Dfunc, 0.0001, D_max)    # # using 0.0001 as lower limit to avoid division by 0
#         print(D_func_int)
        return D_func_int
       
        
    def Tmatrixcalc(self, l1):
        
        """
        Initializes a scatterer object of class pytmatrix.Scatterer
        Make sure particle size and wavelength have the same unit
        Current units: meters
        
        Args:
        l1: upper layer (instance of class Layers) containing the scatterers
        
        Returs:
        Instance of class pytmatrix.Scatterer initialized woth values from vrt.l1 
        
        """

        scatterer = tmatrix.Scatterer(radius = l1.inclusions.a, 
                                      wavelength = self.wavelength, 
                                      m = l1.inclusions.ri, 
                                      axis_ratio = self.l1.inclusions.axratio,
                                      alpha = np.rad2deg(self.l1.inclusions.alpha), 
                                      beta = np.rad2deg(self.l1.inclusions.beta))

        
        # # orientation averaging
        scatterer.orient = orientation.orient_averaged_fixed
        scatterer.or_pdf = orientation.uniform_pdf()
        
         # particle size distribution
        D_med = 2*l1.inclusions.Dmax          # median diameter in m
#         print(D0)
        D_max = l1.inclusions.Dmax        # maximum diameter in m
        Lambda = 1          # some parameter to change for particle size?
        mu = 1
        N = l1.inclusions.nw   
#         N = 1
        scatterer.psd_integrator = PSDIntegrator(D_max = D_max)
        scatterer.psd = ExponentialPSD(N0=N, Lambda=Lambda)
#         scatterer.psd = GammaPSD(D0 = D0, Nw = N, mu=mu)

        # # Use only exponental at the moment. gamma not figured out yet
        self.l1.inclusions.n0 = self.numscatterers_pervolume(scatterer.psd, dist_type="Gamma", nf=N, Lambda=Lambda, mu=mu, D_max = D_max, D_med = D_med)
#         print(self.l1.inclusions.n0 )
        
        return scatterer
          
        
    def PhaseMatrix(self, scatterer, theta_i, phi_i, theta_s, phi_s, poltype):
        
        """
        Calculates Phase matrix for randomly oriented spheroids averaged over orientation 
        (and size distribution hopefully) in the back scattering direction 
        
        Args:
        scatterer: instance of class pytmatrix.Scatterer 
        theta_i: incidence angle in radians
        phi_i: azimuth angle for incident direction in radians
        theta_s: backscatter angle in radians 
        phi_s: azimuth angle for incident direction in radians 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        A 4x4 phase matrix for the spheroidal inclusions/scatterers in the upper layer
        """
        
        scat = scatterer
        geom = (np.rad2deg(theta_i), np.rad2deg(theta_s), np.rad2deg(phi_i), np.rad2deg(phi_s),
                np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))
        scat.set_geometry(geom)

        scat.psd_integrator.geometries = (geom,)
        scat.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
     
        P = self.l1.inclusions.n0 * scat.get_Z()
        
        # # rotating the phase matrix because it seems like the right thing to do
        # # based on Jin 2007, Campbell pr book, Ishimaru 1984 and Mischenko 1996
        
#         if poltype == "lin":
#             B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])  
#         elif poltype == "circ":
#             B = 0.5 * np.matrix([[0, 1.0, 1j, 0.], [1., 0., 0., 1.], [1., 0., 0., -1.], [0, 1.0, -1j, 0.]])  
#         P = np.linalg.multi_dot([B,P,np.linalg.inv(B)])

        return P
    
    def ExtinctionMatrix(self, scatterer, theta, phi):
        
        """
        Calculates Extinction matrix for layer with randomly oriented spheroids averaged over orientation 
        (and size distribution hopefully) in the forward scattering direction using Foldy's approximation
        
        Args:
        scatterer: instance of class pytmatrix.Scatterer 
        theta: incidence angle in radians
        phi: azimuth angle for incident direction in radians
        
        Returns:
        beta: eigenvalues of the extinction matrix
        E: matrix whose columns are the eigenvectors of the extinction matrix
        Einv: inverse of the matrix E
        """
        
        scat = scatterer
        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
                np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))
        scat.set_geometry(geom)
       
        scat.psd_integrator.geometries = (geom,)
        scat.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
      
        SA_dyad = scat.get_S()          # # fvv and fhh appear equal?

        # # Tsang (1985) - eq. 5 in page 139 - attenuation rate matrix
        
        M = (1j * 2 * np.pi * self.l1.inclusions.n0 / self.k) * SA_dyad
#         M = 1j * 2 * np.pi / self.l1.k * SA_dyad

        
        beta = self.StokesEigenValues(M)
        E = self.StokesEigenVectors(M)
        Einv = np.linalg.inv(E)

        return beta, E, Einv
    
            
    def StokesEigenValues(self, M):
        
        """
        Calculate the eigenvalues for the  Extinction matrix 
        
        Args:
        M: A 2x2 matrix computed from the scattering matrix
        
        Returns:
        beta: A numpy array of eigenvalues for the extinction matrix.
        """
        
        r = np.sqrt((M[0,0] - M[1,1])**2 + 4*M[1,0]*M[0,1])
        K1 = self.k - ((1j/2) * (M[0,0] + M[1,1] + r))
        K2 = self.k - ((1j/2) * (M[0,0] + M[1,1] - r))
       
        beta = np.array([2*K1.imag, 1j*np.conj(K2) - 1j*K1, 1j*np.conj(K1) - 1j*K2, 2*K2.imag])
       
        return beta        
    
    def StokesEigenVectors(self, M):
        
        """
        Calculate the eigenvectors for the  Extinction matrix 
        
        Args:
        M: A 2x2 matrix computed from the scattering matrix
        
        Returns:
        E: A numpy array the columns of which are the eigenvectors of the extinction matrix.
        """
        
        # # b1, b2, Mhv, Mvh should be 0 for high frequencies and non-polarizing components
        
        r = np.sqrt((M[0,0] - M[1,1])**2 + 4*M[0,1]*M[1,0])
        b1 = 2 * M[1,0] / (M[0,0] - M[1,1] + r)
        b2 = 2 * M[0,1] / (-M[0,0] + M[1,1] - r)
        
        E = np.matrix([[1, np.conj(b2), b2, np.abs(b2)**2],\
                        [np.abs(b1)**2, b1, np.conj(b1), 1],\
                        [2*b1.real, 1+(b1*np.conj(b2)), 1+(b2*np.conj(b1)), 2*b2.real], \
                        [-2*b1.imag, -1j * (1-(b1*np.conj(b2))), 1j * (1-(b2*np.conj(b1))), 2*b2.imag]])
    
        return E
           
   
    def D(self, beta, theta, z=0.):
        if z == 0.: 
            z = - self.l1.d
        D = np.diag(np.exp(beta * - self.l1.d/ np.cos(theta)))
        return D
    
    
    def integD(self, beta, theta, limits=[0,0], depth=0):
        
        d_int = np.ones_like(beta)
        for i in range(len(beta)):
            d = lambda z: np.exp((beta[i] * (z + depth) / np.cos(theta)))
            d_int[i], err = scipy.integrate.quad(d, limits[0], limits[1])
        
        D_int = np.diag(d_int)
        return D_int   
    
            
   
    def ExtinctionMatrixSimp(self, scatterer, theta, phi, pol):
        
        """
        Calculates a diagonal Extinction matrix for randomly oriented spheres
        in the forward scattering direction using the scattering matrix
        
        Args:
        scatterer: instance of class pytmatrix.Scatterer 
        theta: incidence angle in radians
        phi: azimuth angle for incident direction in radians
        pol: polarization orientation "H" or "V"
        
        Returns:
        K_e: A 4x4 diagonal matrix whose diagonal values are given by the extinction cross section ext_cs
        ext_cs: extinction cross section 
        sca_cs: scattering cross section 
        ssa: single scatter albedo (scat_cs/ext_cs ?)
        """
        
        scat = scatterer
        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
                np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))
        scat.set_geometry(geom)
        
        scat.psd_integrator.geometries = (geom,)
        scat.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
        
        SA_dyad = scat.get_S()
        
        # # use S_vv for vertical pol and S_hh for horizontal pol
        if pol == self.polV: 
            ext_cs = (4 * np.pi / self.k) * SA_dyad[0,0].imag
#             sca_cs = self.l1.inclusions.n0 * 4 * np.pi * (np.abs(SA_dyad[0,0]) ** 2 + np.abs(SA_dyad[0,1]) ** 2)
            
        elif pol == self.polH: 
            ext_cs = (4 * np.pi / self.k) * SA_dyad[1,1].imag
#             sca_cs = self.l1.inclusions.n0 * 4 * np.pi * (np.abs(SA_dyad[1,0]) ** 2 + np.abs(SA_dyad[1,1]) ** 2)
            
        # # make a diagonal matrix
        K_e = np.zeros((4,4))
        np.fill_diagonal(K_e, ext_cs)
        
        print("My Ext CS = ", ext_cs)
        print("My Ext CS mult by n0 = ", ext_cs * self.l1.inclusions.n0)
        
        def ScatteringCS(theta, phi):
#             (scat.phi, scat.thet) = (np.rad2deg(phi), np.rad2deg(theta))
            geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
            np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))
            scat.set_geometry(geom,)

            scat.psd_integrator.geometries = (geom,)
            scat.psd_integrator.init_scatter_table(scatterer, angular_integration=True)
            Z = scat.get_Z()
            if pol == self.polV:            
                scat_int = Z[0,0] + Z[0,1]
            elif pol == self.polH: 
                scat_int = Z[0,0] - Z[0,1]

            return scat_int * np.sin(theta)
        
        sca_cs, err = scipy.integrate.dblquad(ScatteringCS, 0, 2*np.pi, lambda x: 0.0, lambda x: np.pi)
        
   
        if pol == self.polH: cond = True
        elif pol == self.polV: cond = False
        scat.psd_integrator.geometries = (geom,)
        scat.psd_integrator.init_scatter_table(scatterer, angular_integration=True)
        ssa = scatter.ssa(scat, h_pol = cond)

        return K_e, ext_cs, sca_cs, ssa
        
    
    
    def sigma2sigma0(self, sigma):
        
        """
        Computes (unlogarithmic) radar cross section from backscatter coeffiecient
        
        Args: 
        sigma: An array or single value of radar cross-section
        
        Returns:
        sigma0: An array or single value of backscatter coeffiecient
        """
        try:
            if len(sigma) > 1:
                sigma = np.array(sigma)
        except:
            pass

        sigma0 = 10 * np.log10(sigma)   
        return sigma0
        
    
    def sigma02sigma(self, sigma0):
        
        """
        Computes (logarithmic) backscatter coeffiecient from radar cross section
        
        Args: 
        sigma0: An array or single value of backscatter coeffiecient
        
        Returns:
        sigma: An array or single value of radar cross-section        
        """
        
        try:
            if len(sigma0) > 1:
                sigma0 = np.array(sigma0)
        except:
            pass

        sigma = 10 ** (sigma0/10)         
        return sigma
    
    def integrate(self, beta, theta, matrix, limits=[0.,0.], depth=0.):
        n = matrix.shape[0]
        matrix_int = np.zeros_like(matrix)
        
        for i in range(n):
            for j in range(n):
                d = lambda z: np.exp((beta[i] * (z + depth) / np.cos(theta))) * matrix[i,j]
                matrix_int[i,j], err = scipy.integrate.quad(d, limits[0], limits[1])
        return matrix_int
     
    def integtest(self, beta, theta, limits=[0.,0.], depth=0.):
        n = len(beta)
        beta_int = np.zeros((n,n))
        
        for i in range(n):
            d = lambda z: np.exp((beta[i] * (z + depth) / np.cos(theta))) 
            beta_int[i,i], err = scipy.integrate.quad(d, limits[0], limits[1])
        return beta_int
        
    def Mueller_surf(self):
        """
        Compute the real-valued 4x4 Mueller matrix for scattering 
        from the atmosphere - first layer interface.
        """
        ## only returns incoherent scattered power in the backscatter direction
        ## coherent/specular return is zero in the backscatter direction
        return self.R01        
    
    def Mueller_bed(self, scat, poltype):
        
        """
        Compute the real-valued 4x4 Mueller matrix for scattering 
        from the first layer - substrate interface.
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """
        
        # # keeping only the term with the the two coherent transmission matrice (term 2 of C2 in Fa et al. 2011)

        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrix(scat, np.pi - self.l2.theta_i, self.phi_i)
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrix(scat, self.l2.theta_s, self.phi_s)

        
        M_bed = np.linalg.multi_dot([self.T10_coh, E_plus, self.D(beta_plus, self.l2.theta_s), Einv_plus, self.R12, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus, self.T01_coh])
        
#         np.set_printoptions(formatter={'complex_kind': '{:.4f}'.format},suppress = True)
#         pol = self.polH
#         print("Bed scattering")
#         print("Polarization = ", pol)
#         print("Transmitted through = ", self.intensity_breakdown(self.T01_coh, pol))
# #         print(self.T01)
#         print("Attenuated to bottom = ", self.intensity_breakdown(np.linalg.multi_dot([E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus, self.T01_coh]), pol))
# #         print(np.linalg.multi_dot([E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus]))
#         print("Reflected at bottom = ", self.intensity_breakdown(np.linalg.multi_dot([self.R12, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus, self.T01_coh]), pol))
#         print("Attenuated back to top = ", self.intensity_breakdown(np.linalg.multi_dot([E_plus, self.D(beta_plus, self.l2.theta_s), Einv_plus, self.R12, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus, self.T01_coh]), pol))
# #         print(np.linalg.multi_dot([E_plus, self.D(beta_plus, self.l2.theta_s), Einv_plus]))
#         print("Transmitted back out = ", self.intensity_breakdown(np.linalg.multi_dot([self.T10_coh, E_plus, self.D(beta_plus, self.l2.theta_s), Einv_plus, self.R12, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus, self.T01_coh]), pol))
        
        return M_bed
                
    
    def Mueller_bedvol(self, scat, poltype):
        
        """
        Compute the real-valued 4x4 Mueller matrix for scattering from the 
        inclusions in the first layer (2) and first layer - substrate interface (1).
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """
        
        # # theta and phi for inclusion / scatterer
        # # theta_i = theta_s = np.pi/2 - theta_i of incoming radiation; phi_i = phi_s = 0

        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrix(scat, np.pi - self.l2.theta_i, self.phi_i)
        beta_plus1, E_plus1, Einv_plus1 = self.ExtinctionMatrix(scat, self.l2.theta_s, self.phi_i)
        beta_plus2, E_plus2, Einv_plus2 = self.ExtinctionMatrix(scat, self.l2.theta_s, self.phi_s)
        

        M_bedvol = np.linalg.multi_dot([Einv_plus1, self.R12_coh, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus, self.T01_coh])
        M_bedvol =self.integrate(beta_plus1, self.l2.theta_s, M_bedvol, [-self.l1.d, 0], - self.l1.d)
        M_bedvol = np.linalg.multi_dot([Einv_plus2 , self.PhaseMatrix(scat, np.pi/2 - self.l2.theta_s, self.phi_i, np.pi/2 - self.l2.theta_s, self.phi_i, poltype),E_plus1, M_bedvol])
        M_bedvol = self.integrate(beta_plus2, self.l2.theta_s, M_bedvol, [-self.l1.d, 0])
        M_bedvol = np.linalg.multi_dot([self.T10_coh / np.cos(self.l2.theta_s), E_plus2, M_bedvol])
        
#         print("bedvol")
#         print(M_bedvol)
        
#         A = self.ExtinctionMatrixComb(beta_plus2, self.l2.theta_s, beta_plus1, self.l2.theta_i)

#         A = np.zeros((4,4), dtype=complex)
#         for i in range(A.shape[0]):
#             for j in range(A.shape[1]):
#                 A[i,j] = (np.exp(beta_plus2[i] * -self.l1.d / np.cos(self.l2.theta_s)) - np.exp(beta_plus1[j] * -self.l1.d / np.cos(self.l2.theta_s))) / \
#                 (-beta_plus2[i]/np.cos(self.l2.theta_i) + (beta_plus1[j]/np.cos(self.l2.theta_i)))
        
        
#         M_bedvol =  np.linalg.multi_dot([A * np.linalg.multi_dot([Einv_plus2 , self.PhaseMatrix(scat, np.pi/2 - self.l2.theta_s, self.phi_i, np.pi/2 - self.l2.theta_s, self.phi_i,  poltype) ,E_plus1]), Einv_plus1, self.R12_coh, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus,self.T01])
        
#         M_bedvol =  np.linalg.multi_dot([self.T10 / np.cos(self.l2.theta_s), E_plus2, M_bedvol])


#         M_bedvol = np.linalg.multi_dot([self.T10_cof / np.cos(self.l2.theta_s), E_plus2,\
#                                         Einv_plus2 , self.PhaseMatrix(scat, np.pi/2 - self.l2.theta_s, self.phi_i, np.pi/2 - self.l2.theta_s, self.phi_i, poltype),E_plus1,\
#                                         A, Einv_plus1, self.R12_coh, E_minus, self.D(beta_minus, self.l2.theta_i), Einv_minus,self.T01_coh])  
        
        return M_bedvol
    
    def Mueller_volbed(self, scat, poltype):

        """
        Compute the real-valued 4x4 Mueller matrix for scattering from the 
        inclusions in the first layer (1) and first layer - substrate interface (2).
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """
        # # theta and phi for inclusion / scatterer
        # # theta_i = theta_s = np.pi/2 - theta_i of incoming radiation; phi_i = phi_s = 0
        
        
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrix(scat, self.l2.theta_s, self.phi_s)
        beta_minus2, E_minus2, Einv_minus2 = self.ExtinctionMatrix(scat, np.pi - self.l2.theta_i, self.phi_s)
        beta_minus1, E_minus1, Einv_minus1 = self.ExtinctionMatrix(scat, np.pi - self.l2.theta_i, self.phi_i)
        
        M_volbed = np.linalg.multi_dot([Einv_minus1, self.T01_coh])
        M_volbed = self.integrate(beta_minus1, self.l2.theta_i, M_volbed, [-self.l1.d, 0])
        M_volbed = np.linalg.multi_dot([Einv_minus2, self.PhaseMatrix(scat, np.pi - (np.pi/2 - self.l1.theta_t), self.phi_i, np.pi - (np.pi/2 - self.l1.theta_t), self.phi_i,  poltype), E_minus1, M_volbed])
        M_volbed = self.integrate(beta_minus2, self.l2.theta_i, M_volbed, [-self.l1.d, 0], -self.l1.d)
        M_volbed = np.linalg.multi_dot([self.T10_coh, E_plus, self.D(beta_plus, self.l2.theta_s), Einv_plus, self.R12_coh / np.cos(self.l2.theta_s), E_minus2 , M_volbed])
        
#         print("vol_bed")
#         print(M_volbed)
        
# #         A = self.ExtinctionMatrixComb(beta_minus2, self.l2.theta_s, beta_minus1, self.l2.theta_i)
        
#         A = np.zeros((4,4), dtype=complex)
#         for i in range(A.shape[0]):
#             for j in range(A.shape[1]):
#                 A[i,j] = (np.exp(beta_minus2[i] * -self.l1.d / np.cos(self.l1.theta_t)) - np.exp(beta_minus1[j] * -self.l1.d / np.cos(self.l1.theta_t))) / \
#                 (-beta_minus2[i]/np.cos(self.l1.theta_t) + (beta_minus1[j]/np.cos(self.l1.theta_t)))
                                      
                    
#         M_volbed = np.linalg.multi_dot([A * np.linalg.multi_dot([Einv_minus2, self.PhaseMatrix(scat, np.pi - (np.pi/2 - self.l1.theta_t), self.phi_i, np.pi - (np.pi/2 - self.l1.theta_t), self.phi_s,  poltype), E_minus1]), Einv_minus1, self.T01_coh])
        
#         M_volbed = np.linalg.multi_dot([self.T10_coh / np.cos(self.l2.theta_s), E_plus, self.D(beta_plus, self.l2.theta_s), Einv_plus, self.R12_coh, E_minus2 , M_volbed])
        
        return M_volbed
    
    def Mueller_vol(self, scat, poltype):  
        
        """
        Compute the real-valued 4x4 Mueller matrix for scattering 
        from the inclusions in the first layer.
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """

        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrix(scat, np.pi - self.l1.theta_t, self.phi_i)
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrix(scat, self.l2.theta_s, self.phi_s)
        
        M_vol = np.linalg.multi_dot([Einv_minus, self.T01_coh])
        M_vol = self.integrate(beta_minus, self.l2.theta_s, M_vol, [-self.l1.d, 0]) 
        M_vol = np.linalg.multi_dot([Einv_plus, self.PhaseMatrix(scat, np.pi - self.l1.theta_t, self.phi_i, self.l1.theta_t, self.phi_s,  poltype), E_minus, M_vol])
        M_vol = self.integrate(beta_plus, self.l2.theta_s, M_vol, [-self.l1.d, 0]) 
        M_vol = np.linalg.multi_dot([self.T10_coh / np.cos(self.l1.theta_i), E_plus, M_vol])
        
#         A = np.zeros((4,4), dtype=complex)
#         gp1 = np.linalg.multi_dot([E_plus, Einv_plus, self.PhaseMatrix(scat, np.pi - self.l1.theta_t, self.phi_i, self.l1.theta_t, self.phi_s,  poltype), E_minus])
        
#         for i in range(A.shape[0]):
#             for j in range(A.shape[1]):
#                 A[i,j] = ((1 - np.exp((beta_plus[i]*-self.l1.d/np.cos(self.l2.theta_s)) - (beta_minus[j]*-self.l1.d/np.cos(self.l1.theta_t)))) / (beta_plus[i]/np.cos(self.l2.theta_s) + (beta_minus[j]/np.cos(self.l1.theta_t)))) 
                
#         gp2 = A * np.array(Einv_minus)
        
# #         M_med = np.linalg.multi_dot([gp1, gp2])
#         M_med = np.array(gp1) *  np.array(gp2)
#         M_vol = (1/ np.cos(self.l1.theta_i)) * np.linalg.multi_dot([self.T10_coh, M_med, self.T01_coh])

#         print("Volume scattering")
#         print("Polarization = ", pol)
#         M_vol = self.T01
#         print("Transmitted through = ", self.intensity_breakdown(M_vol, pol))
#         M_vol = np.linalg.multi_dot([Einv_minus, M_vol])
#         M_vol = self.integrate(beta_minus, self.l2.theta_s, M_vol, [-self.l1.d, 0]) 
#         M_vol = np.linalg.multi_dot([E_minus, M_vol])
#         print("Attenuated to bottom = ", self.intensity_breakdown(M_vol, pol))
#         M_vol = np.linalg.multi_dot([self.PhaseMatrix(scat, np.pi - self.l1.theta_t, self.phi_i, self.l1.theta_t, self.phi_s, poltype), M_vol])
#         print("Scattered by inclusion = ", self.intensity_breakdown(M_vol, pol))
#         M_vol = np.linalg.multi_dot([Einv_plus, M_vol])
#         M_vol = self.integrate(beta_plus, self.l2.theta_s, M_vol, [-self.l1.d, 0]) 
#         M_vol = np.linalg.multi_dot([E_plus, M_vol])
#         print("Attenuated back to top = ", self.intensity_breakdown(M_vol, pol))
#         M_vol = np.linalg.multi_dot([self.T10 / np.cos(self.l2.theta_s), M_vol])
#         print("Transmitted back out = ", self.intensity_breakdown(M_vol, pol))

#         print("vol")
#         print(M_vol)
        
        return M_vol
    
    def intensity_breakdown(self, M, pol):
        if pol == self.polH:
            I_i = np.array([0, 1., 0., 0.]).reshape((4,1))
            I_s = M * I_i
            sigma = 4 * np.pi * np.cos(self.theta_s) * (I_s[1,0]/ I_i[1,0])
        elif pol == self.polV:
            I_i = np.array([1., 0, 0., 0.]).reshape((4,1))
            I_s = M * I_i
            sigma = 4 * np.pi * np.cos(self.theta_s) * (I_s[0,0]/ I_i[0,0])
            
        return sigma
        
    def Mueller_lin(self, tm, pol):
        """
        Compute the Mueller matrix solution for VRT for linear polarization (H and V)
        
        Args:
        tm: instance of class pytmatrix.Scatterer 
        pol: polarization orientation "H" or "V" 
        
        Returns:
        sigma0: A 5x1 array containing the backscatter coefficient associated with 
                (0) Total BS, (1) Surface BS, (2) Subsurface BS, (3) Volume BS, and
                (4) Voume-subsurface BS
        DLP:  5x1 array containing the degree of linear polarization
        """
        
        if pol == self.polH:
            I_i = np.array([0., 1., 0., 0.]).reshape((4,1))
            idx = 1
        elif pol == self.polV:
            I_i = np.array([1., 0., 0., 0.]).reshape((4,1))
            idx = 0

        poltype = "lin"
        B =np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0.,0.,1.,0.],[0.,0.,0.,1.]])          # # modified stokes matrix rotation
        Binv = np.linalg.inv(B)
        # # surface backscatter
        M_surf = self.R01
        
        [M_bed, M_vol, M_bedvol, M_volbed] = [self.Mueller_bed(tm, poltype), self.Mueller_vol(tm, poltype), self.Mueller_bedvol(tm, poltype), self.Mueller_volbed(tm, poltype)]
        M_total = M_surf + M_bed + M_vol + M_bedvol + M_volbed
        M = np.array([M_total, M_surf, M_bed, M_vol, M_bedvol+M_volbed]) 
#         I_s = np.array([np.linalg.multi_dot([B,M_total,Binv,I_i]), \
#                         np.linalg.multi_dot([B,M_surf,Binv,I_i]), \
#                         np.linalg.multi_dot([B,M_bed,Binv,I_i]), \
#                         np.linalg.multi_dot([B,M_vol,Binv,I_i]), \
#                         np.linalg.multi_dot([B,M_bedvol+M_volbed,Binv,I_i])])
# #         print("I_s lin = ", I_s[0])
       
#         # Backscattering cross-section not normalized to area
#         if pol == self.polV: 
#             sigma_0 = 4 * np.pi * np.cos(self.theta_s) * I_s[:,0,0]
#         elif pol == self.polH: 
#             sigma_0 = 4 * np.pi * np.cos(self.theta_s) * I_s[:,1,0]
            
        I_s = np.zeros((len(M), 4, 1))
        for i in range(len(M)):
            M[i] = np.linalg.multi_dot([B, M[i], np.linalg.inv(Binv)])
            I_s[i] = np.linalg.multi_dot([M[i], I_i])
        
        if pol == self.polV: 
            sigma_0 = 4 * np.pi * np.cos(self.theta_s) * (M[:,0,0] + 2*M[:,0,1] + M[:,1,1]) / 2
        elif pol == self.polH: 
            sigma_0 = 4 * np.pi * np.cos(self.theta_s) * (M[:,0,0] - 2*M[:,0,1] + M[:,1,1]) / 2
        

#         DLP = np.sqrt(I_s[:,1,0]**2 + I_s[:,2,0]**2) / I_s[:,0,0]
        DLP = 0

        return sigma_0, DLP
    
    def Mueller_circ(self, tm, pol):
        
        """
        Compute the Mueller matrix solution for VRT for circulare polarization (LC and RC)
        
        Args:
        tm: instance of class pytmatrix.Scatterer 
        pol: polarization orientation "LC" or "RC" 
        
        Returns:
        CPR:  5x1 array containing the circular polarization ratio
        """
        if pol == "RC":
#             I_i = np.array([0., 1., 0., 0.]).reshape((4,1))
#             I_i = np.array([0.5, 0.5, 0., -1.]).reshape((4,1))
            I_i = np.array([1., 0., 0., -1.]).reshape((4,1))
        elif pol == "LC":
#             I_i = np.array([0., 0., 1., 0.]).reshape((4,1))
#             I_i = np.array([0.5, 0.5, 0., 1.]).reshape((4,1))
            I_i = np.array([1., 0., 0., 1.]).reshape((4,1))
            
        poltype = "circ"
        # # surface backscatter
        M_surf = self.R01
        
        [M_bed, M_vol, M_bedvol, M_volbed] = [self.Mueller_bed(tm, poltype), self.Mueller_vol(tm, poltype), self.Mueller_bedvol(tm, poltype), self.Mueller_volbed(tm, poltype)]
        M_total = M_surf + M_bed + M_vol + M_bedvol + M_volbed
        M = np.array([M_total, M_surf, M_bed, M_vol, M_bedvol+M_volbed]) 
#         print(M_surf)
#         I_s = np.array([np.linalg.multi_dot([M_surf,I_i]), \
#                         np.linalg.multi_dot([M_surf,I_i]), \
#                         np.linalg.multi_dot([M_bed,I_i]), \
#                         np.linalg.multi_dot([M_vol,I_i]), \
#                         np.linalg.multi_dot([M_bedvol+M_volbed,I_i])])
        
# #         CPR = (I_s[:,0,0] + I_s[:,1,0] - I_s[:,3,0])/(I_s[:,0,0] + I_s[:,1,0] + I_s[:,3,0])
#         CPR = (I_s[:,0,0] - I_s[:,3,0])/(I_s[:,0,0] + I_s[:,3,0])
        
        
#         I_s = np.zeros((len(M), 4, 1))
        
#         A = 0.5 * np.matrix([[0., 1., 1j, 0.], [1.,0.,0.,1.], [1.,0.,0.,-1.],[0.,1., -1j,0.]])
# #         A = 0.5 * np.matrix([[1., -1., -1j, 0.], [1.,1.,0.,-1.], [1.,1.,0.,1.],[1.,-1., 1j,0.]])
        
#         for i in range(len(M)):
#             M[i] = np.linalg.multi_dot([A, M[i], np.linalg.inv(A)])
#             I_s[i] = np.linalg.multi_dot([M[i], I_i])
            
        CPR = np.abs((M[:,0,0] - M[:,3,3]) / np.abs(M[:,0,0] + M[:,3,3])) 
            
# #         print(M[1])
# #         print(I_s)
        
#         # Backscattering cross-section not normalized to area
#         if pol == "RC": 
#             sigma_sc = 4 * np.pi * np.cos(self.theta_s) * I_s[:,1,0]
#             sigma_oc = 4 * np.pi * np.cos(self.theta_s) * I_s[:,2,0]
#         elif pol == "LC": 
#             sigma_sc = 4 * np.pi * np.cos(self.theta_s) * I_s[:,2,0]
#             sigma_oc = 4 * np.pi * np.cos(self.theta_s) * I_s[:,1,0]
        
# #         CPR = sigma_sc / sigma_oc
# #         CPR [0] = CPR[1]
#         CPR = M[:,0,0]- M[:,3,3] / M[:,0,0]+ M[:,3,3]
#         CPR [0] = CPR[1]
        return CPR

        
    def I2EM(self, l1, l2):
        """
        Compute the HH and VV rough surface backscatter using 
        Improved Integral Equation Method (I2EM).
        
        Args:
        l1: upper medium (instance of class Layers)
        l2: lower medium (instance of class Layers)
        
        Returns:
        sigmavv_new: Backscatter coeffecient for VV polarization
        sigmahh_new: Backscatter coeffecient for HH polarization
        sigmavv_t: Transmitted scattering coeffecient for VV polarization
        sigmahh_t: Transmitted scattering coeffecient for HH polarization
   
        """
        theta_i = np.rad2deg(l2.theta_i)
        
        eng = matlab.engine.start_matlab()
        sigma0vv, sigma0hh, sigma0hv = eng.I2EM_Backscatter_model(self.nu/1e9, l2.upperks, l2.corrlen, theta_i.item(), l2.eps/l1.eps, 1, 0, nargout=3)           # using .item() to convert from numpy float64 to python scalar
        svv, shh, svh = self.sigma02sigma([sigma0vv, sigma0hh, sigma0hv])
                
        interface = RoughSurface(self.wavelength, l1.eps, l2.eps, l2.upperks, l2.corrlen, autocorr_fn = "exponential", theta_i = l2.theta_i, theta_s = l2.theta_s)
        sigmavv_new, sigmahh_new = interface.Copol_BSC()
        
        sigmavv_t, sigmahh_t = interface.Copol_BSC_trans()

        print("i2em ", sigma0vv)


#         return svv, sigmavv_new, shh, sigmahh_new, sigmavv_t, sigmahh_t

    def I2EM_emissivity(self, l1, l2):
        
        """
        Compute the H and V rough surface emissivity using rough surface 
        reflectivity computed by the Improved Integral Equation Method.
        
        Args:
        l1: upper medium (instance of class Layers)
        l2: lower medium (instance of class Layers)
        
        Returns:
        e_v: Rough surface emissivity in V polarization
        e_h: Rough surface emissivity in H polarization
        """
        
        theta_i = np.rad2deg(l2.theta_i)
        
        # # if float is not iterable error rises, check if the matlab function returns are separated by commas
        eng = matlab.engine.start_matlab()
        e_v, e_h = eng.I2EM_Emissivity_model(self.nu/1e9, l2.upperks, l2.corrlen, theta_i.item(), l2.eps/l1.eps, 1, nargout=2)           # using .item() to convert from numpy float64 to python scalar

        return e_v, e_h
    
    
    def VRTsolver(self):
        """
        Primary function that executes the 1st order VRT scattering model
        
        Returns: A list of sigma0_vv, sigma0_hh, cpr, e_v, e_h
        sigma0vv: A 5x1 array containing the backscatter coefficient associated with 
                (0) Total BS, (1) Surface BS, (2) Subsurface BS, (3) Volume BS, and
                (4) Voume-subsurface BS
        sigma0hh: A 5x1 array containing the backscatter coefficient associated with 
                the same mechanisms as sigma0vv
        cpr:  5x1 array containing the circular polarization ratio associated with 
              the same mechanisms as sigma0vv
        e_v: Emissivity in V polarization
        e_h: Emissivity in H polarization
        """
        
#         np.set_printoptions(formatter={'complex_kind': '{:.8}'.format},suppress = True)
        
        self.setGeometricOptics()
        self.setLayerProperties()
   
                        
        # # Create a rough surface object for estimating backscatter and transmitted scatter coefficient
        self.i01 = RoughSurface(self.wavelength, self.atm.eps, self.l1.eps, self.l1.upperks, self.l1.corrlen, autocorr_fn = "exponential", theta_i = self.l1.theta_i, theta_s = self.l1.theta_s)
        self.i10 = RoughSurface(self.wavelength, self.l1.eps, self.atm.eps, self.l1.upperks, self.l1.corrlen, autocorr_fn = "exponential", theta_i = self.l2.theta_s, theta_s = self.l1.theta_s)
        self.i12 = RoughSurface(self.wavelength, self.l1.eps, self.l2.eps, self.l2.upperks, self.l2.corrlen, autocorr_fn = "exponential", theta_i = self.l2.theta_i, theta_s = self.l2.theta_s)
        
#         self.R01 = self.NonCoherentRefMatrix(self.atm, self.l1, self.l1.theta_i)
#         self.R12 = self.NonCoherentRefMatrix(self.l1, self.l2, self.l1.theta_t)
        
        self.R01 = self.NonCoherentRefMatrix(self.i01, self.l1.theta_i)
        self.R12 = self.NonCoherentRefMatrix(self.i12, self.l1.theta_t)
        
        self.R01_coh = self.CoherentRefMatrix(self.atm, self.l1, self.l1.theta_i)
        self.R12_coh = self.CoherentRefMatrix(self.l1, self.l2, self.l1.theta_t)
                                  
        # # adding a factor in front of transmission matrix based on equation 11.61b in Ulaby big book
                                          
        self.T01 = self.NonCoherentTransMatrix(self.i01, self.l1.theta_i)
        self.T10 = self.NonCoherentTransMatrix(self.i10, self.l2.theta_s)
        
        self.T01_coh = self.CoherentTransMatrix(self.atm, self.l1, self.l1.theta_i)
        self.T10_coh = self.CoherentTransMatrix(self.l1, self.atm, self.l2.theta_s)

        # # Create a T-matrix instance for phase matrix and extinction matrix calculations
        tm = self.Tmatrixcalc(self.l1)
  
        # # VRT
        
        sigma_hh, DLP = self.Mueller_lin(tm, self.polH)
        sigma_vv, _ = self.Mueller_lin(tm, self.polV)
        
#         cpr = self.Mueller_circ(tm, "LC")
        cpr = np.zeros(5)

        
        sigma0_hh, sigma0_vv = self.sigma2sigma0([sigma_hh, sigma_vv])
#         e_v, e_h = self.RT0_emission(tm)
        e_v = 0
        e_h = 0

        
        return [sigma0_vv, sigma0_hh, cpr, e_v, e_h]
        
        # # only for plotting fresnel coefficients
#         refh,transh,rh,th = self.FresnelCoefH(self.atm, self.l1, self.theta_i)
#         refv,transv,rv,tv = self.FresnelCoefV(self.atm, self.l1, self.theta_i)
#         return [sigma0_vv, sigma0_hh, CPR, refh, refv, transh, transv, rh, rv, th, tv]
        
#         # # only for plotting surface backscatter
#         svv1, svv2, shh1, shh2, stvv1, sthh1, stvv2, sthh2 = self.reftranscomp()
#         return svv1, svv2, shh1, shh2, stvv1, sthh1, stvv2, sthh2
        

    def RT0_emission(self, scat):
        """
        Primary function that executes the oth order VRT emission model
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        
        Returns:
        e_v: Emissivity in V polarization
        e_h: Emissivity in H polarization
        """
        
        #np.set_printoptions(formatter={'complex_kind': '{:.8}'.format},suppress = True)
        
        self.setGeometricOptics()
        self.setLayerProperties()
        
        beta, E, Einv = self.ExtinctionMatrix(scat, self.l2.theta_s, self.phi_s)
        trans = np.linalg.multi_dot([E, self.D(beta, self.l2.theta_s), Einv])

        print("SSA calculation")
        
        _, ke_v, ks_v, ssa_v = self.ExtinctionMatrixSimp(scat, self.theta_s, self.phi_s, self.polV)
        _, ke_h, ks_h, ssa_h = self.ExtinctionMatrixSimp(scat, self.theta_s, self.phi_s, self.polH)
        
        print("pyt albedo = ", ssa_h, ssa_v)
        
        
        ssa_h = ks_h/ke_h
        ssa_v = ks_v/ke_v

        print("My albedo = ", ssa_h, ssa_v)
        
        a_h = 1-ssa_h
        a_v = 1-ssa_v
        
        e01_v, e01_h = self.I2EM_emissivity(self.atm, self.l1)
        e12_v, e12_h = self.I2EM_emissivity(self.l1, self.l2)
        
        gamma01_v = 1 - e01_v
        gamma01_h = 1 - e01_h
        gamma12_v = 1 - e12_v
        gamma12_h = 1 - e12_h
        
        e_v = ((1 - gamma01_v) / (1 - gamma01_v * gamma12_v * trans[0,0]**2)) * \
            ((1 + gamma12_v*trans[0,0]) * (1-a_v) * (1 - trans[0,0]) + (1 - gamma12_v) * trans[0,0])
        
        e_h = ((1 - gamma01_h) / (1 - gamma01_h * gamma12_h * trans[1,1]**2)) * \
            ((1 + gamma12_h*trans[1,1]) * (1-a_h) * (1 - trans[1,1]) + (1 - gamma12_h) * trans[1,1])
        
        return e_v, e_h

               

    def reftranscomp(self):
        """
        Temporary method for testing the working of self-written I2EM 
        woth Ulaby's I2EM and Fresnel reflection/transmission
        """
        
        # # Create a rough surface object for estimating backscatter and transmitted scatter coefficient
        i01 = RoughSurface(self.wavelength, self.atm.eps, self.l1.eps, self.l1.upperks, self.l1.corrlen, autocorr_fn = "exponential", theta_i = self.l1.theta_i, theta_s = self.l1.theta_s)
        i10 = RoughSurface(self.wavelength, self.l1.eps, self.atm.eps, self.l1.upperks, self.l1.corrlen, autocorr_fn = "exponential", theta_i = self.l2.theta_s, theta_s = self.l1.theta_s)
        i12 = RoughSurface(self.wavelength, self.l1.eps, self.l2.eps, self.l2.upperks, self.l2.corrlen, autocorr_fn = "exponential", theta_i = self.l2.theta_i, theta_s = self.l2.theta_s)
        svv, shh = self.NonCoherentRefMatrix(i01)
        svv_c, shh_c = self.CoherentRefMatrix(self.atm, self.l1, self.l1.theta_i)                        
        # # adding a factor in front of transmission matrix based on equation 11.61b in Ulaby big book                                 
        stvv, sthh = self.NonCoherentTransMatrix(i01)
        stvv_c, sthh_c = self.CoherentTransMatrix(self.atm, self.l1, self.l1.theta_i)
        
        return svv, svv_c, shh, shh_c, stvv, stvv_c, sthh, sthh_c
        
    def VRTsolver_depth(self, value):
        self.l1.d = float(value)
        return self.VRTsolver()
        
    def VRTsolver_eps1(self, value):
        self.l1.eps = complex(value)
        return self.VRTsolver()
        
    def VRTsolver_eps2(self, value):
        self.l2.eps = complex(value)
        return self.VRTsolver()
        
    def VRTsolver_epsscatterer(self, value):
        self.l1.inclusions.eps = complex(value)
        return self.VRTsolver()
        
    def VRTsolver_scatterershape(self, value):
        self.l1.inclusions.axratio = float(value)
        return self.VRTsolver()
        
    def VRTsolver_scatterernumconc(self, value):
        self.l1.inclusions.nw = float(value)
        return self.VRTsolver()
    
    def VRTsolver_scatterersize(self, value):
        self.l1.inclusions.Dmax = float(value)
        return self.VRTsolver()
        
    def VRTsolver_emrough(self, value):
        self.l1.upperks = float(value)
        return self.VRTsolver()
    
    def VRTsolver_emroughsub(self, value):
        self.l2.upperks = float(value)
        return self.VRTsolver()
    
    def VRTsolver_incidence(self, value):
        self.theta_i = np.deg2rad(value)
        return self.VRTsolver()            
