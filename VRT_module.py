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

if poltype == "lin":
 B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])  
elif poltype == "circ":
 B = 0.5 * np.matrix([[0, 1.0, 1j, 0.], [1., 0., 0., 1.], [1., 0., 0., -1.], [0, 1.0, -1j, 0.]])
    
Radar CS backscatter coefficient: 
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
import pandas as pd

import matlab.engine

from pytmatrix import tmatrix, orientation, scatter
from pytmatrix.psd import PSDIntegrator, GammaPSD, ExponentialPSD

import radar_inst as radar
#from Layers import Layers
from RoughSurface import RoughSurface
import FresnelCoefficients as Fresnel


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
#         self.theta_s = self.theta_i
       
        # Layers
        self.atm = layers[0]
        self.l1= layers[1]
        self.l2 = layers[2]
 
    
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
        
#         # # set upper surface
#         self.i01 = RoughSurface(self.wavelength, self.atm.eps, self.l1.eps, self.l1.upperks, self.l1.corrlen, autocorr_fn = "exponential", theta_i = self.l1.theta_i, theta_s = self.l1.theta_s)
#         self.i10 = RoughSurface(self.wavelength, self.l1.eps, self.atm.eps, self.l1.upperks, self.l1.corrlen, autocorr_fn = "exponential", theta_i = self.l2.theta_s, theta_s = self.l1.theta_s)
        
        
#         # # calculate coherent transmittivity matrices
#         self.T01_coh = self.CoherentTransMatrix(self.atm, self.l1, self.l1.theta_i)
#         self.T10_coh = self.CoherentTransMatrix(self.l1, self.atm, self.l2.theta_s)

#         # # set bottom surface
#         if self.l2 != None:
#             self.i12 = RoughSurface(self.wavelength, self.l1.eps, self.l2.eps, self.l2.upperks, self.l2.corrlen, autocorr_fn = "exponential", theta_i = self.l2.theta_i, theta_s = self.l2.theta_s)
            
    
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
#         self.l1.theta_t = self.TransmissionAngle(self.theta_i,self.atm.eps, self.l1.eps)
        self.l2.theta_i = self.l1.theta_t
        self.l2.theta_s = self.l2.theta_i
        
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
        
    def Mueller2sigma(self, M):
        # # multiply by 4*pi*cos theta if needed
        svv = 4 * np.pi * (M[0,0] + 2*M[0,1] + M[1,1]) 
        shh = 4 * np.pi * (M[0,0] - 2*M[0,1] + M[1,1]) 
        return svv, shh
    
    def sigma2Mueller(self, svv, shh, svh=0):
        
        shv = - svh
        s11 = svv / 4 / np.pi
        s22 = shh / 4 / np.pi
        s12 = svh / 4 / np.pi
        s21 = shv / 4 / np.pi
        
        # # Mueller matrix implementation
        m11 = s11 + s22 + 2*s21  
        m12 = s11- s22
        m13 = 0
        m14 = 0
        m22 = s11 + s22 - 2*s21
        m23 = 0
        m24 = 0
        
        
        # # check m33 and m34 and check if s12 or s21 is negative
#         m33 = 2*shv
        m33 = 2* ((s11*s22)**0.5 + s12**0.5)
        m34 = 0
#         m44 = 2*shv
        m44 = 2* ((s11*s22)**0.5 + s12**0.5)
        
        M = np.array([[m11,m12,m13,m14], [m12,m22,m23,m24], [m13, m23, m33, m34], [-m14, -m24, -m34, m44]]) / 4
        return M
    
    def Mueller2cpr(self,M):
        cpr = (M[0,0]+ 2*M[0,3]+M[3,3]) / (M[0,0]-M[3,3])
        return cpr
    
    def sigma2cpr(self, svv, shh, svh):
        cpr = (svv+shh+4*svh)/(svv+shh)       
        return cpr       
        
    
    def TransmissionAngle(self, thetai, eps1, eps2):
        """
        Compute the angle of transmisssion (rad) from one medium to another.
        
        Args:
        l1: upper medium (instance of class RoughSurface)
        l2: lower medium (instance of class RoughSurface)
        
        Returns:
        Angle of transmission (in radians) from l1 to l2
        """
        mu1 = np.cos(thetai)
        n = np.sqrt(eps2/eps1)
        mu2 = np.sqrt(1.0 - ((1.0 - mu1**2) / n**2)).real
        thetat = np.arccos(mu2)
        return thetat

           
    def CoherentTransMatrix(self, eps1, eps2, ks, thetai_rad):
        
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

        mu1 = np.cos(thetai_rad)
        n = np.sqrt(eps2/eps1)
        mu2 = np.sqrt(1.0 - (1.0 - mu1**2) / n**2).real

        refh,transh,rh,th = Fresnel.FresnelH(eps1, eps2, thetai_rad)
        refv,transv,rv,tv = Fresnel.FresnelV(eps1, eps2, thetai_rad)

        K_i = self.k * np.sqrt(eps1.real) * mu1
        K_t = self.k * np.sqrt(eps2 - ((1 - mu1**2) * eps1)).real
        loss_factor = np.exp(-ks**2 * (K_t - K_i)**2)
               
        # # Modified Mueller matrix convention - not following the refractive index factor in eq 11.61b in Ulaby big book
#         fact = (n**3 * mu2 / mu1).real
        m11 = np.abs(transv) ** 2
        m22 = np.abs(transh) ** 2
        m33 = (transv * np.conj(transh)).real
        m43 = (transv * np.conj(transh)).imag
#         T = np.matrix([[m11, 0., 0., 0.], [0., m22, 0., 0.], [0., 0., m33 * mu2/mu1, -m43* mu2/mu1], [0., 0., m43* mu2/mu1, m33* mu2/mu1]])
    
        # # Mueller matrix convention
        T = 0.5 * np.matrix([[tv+th, tv-th, 0., 0.], [tv-th, tv+th, 0., 0.], [0., 0., 2*(tv*th)**0.5, 0.], [0., 0., 0., 2*(tv*th)**0.5]])
        T_coh = loss_factor * T
        
        return T_coh
    
    def CoherentRefMatrix(self, eps1, eps2, ks, thetai_rad):
        
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
        
        mu = np.cos(thetai_rad)
        
        refh,transh,rh,th = Fresnel.FresnelH(eps1, eps2, thetai_rad)
        refv,transv,rv,tv = Fresnel.FresnelV(eps1, eps2, thetai_rad)

        K_s2 = self.k ** 2 * (eps1.real ** 2 + eps1.imag ** 2)                # square of the wavenumber in the scattered medium
        
        loss_factor = np.exp(-2 * ks**2 * K_s2 * mu**2)
        
        # # Modified Mueller matrix convention
        r11 = rh
        r22 = rv
        r33 = (refv * np.conj(refh)).real
        r43 = (refv * np.conj(refh)).imag
#         R = np.matrix([[r11, 0., 0., 0.], [0., r22, 0., 0.], [0., 0., r33, -r43], [0., 0., r43, r33]])
        
        # Mueller matrix convention
        R = 0.5 * np.matrix([[rv+rh, rv-rh, 0., 0.], [rv-rh, rv+rh, 0., 0.], [0., 0., 2*(rv*rh)**0.5, 0.], [0., 0., 0., 2*(rv*rh)**0.5]])

        R_coh = loss_factor * R

        return R_coh
    

    def numscatterers_pervolume(self, psd, dist_type="Gamma", nf=1, Lambda=1, mu=.01, D_max = 0.06, D_med = 1):
        
        """ 
        Compute the integrated total number concentration of particles in the upper layer.
        
        # # CURRENTLY DOESN'T WORK. NEEDS FIXING. 
        
        Args:
        psd: particle size distribution (instance of pytmatrix.psd class)
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
        return D_func_int
       
        
    def Tmatrixcalc(self, radius, ri, axis_ratio, alpha, beta, **psdargs):
        
        """
        Initializes a scatterer object of class pytmatrix.Scatterer
        Make sure particle size and wavelength have the same unit
        Current units: meters
        
        Args:
        l1: upper layer (instance of class Layers) containing the scatterers
        
        Returs:
        Instance of class pytmatrix.Scatterer initialized woth values from vrt.l1 
        
        """

        scatterer = tmatrix.Scatterer(radius = radius,
                                      wavelength = self.wavelength, 
                                      m = ri, 
                                      axis_ratio = axis_ratio,
                                      alpha = alpha, 
                                      beta = beta)

        
        # # orientation averaging
        scatterer.orient = orientation.orient_averaged_fixed
        scatterer.or_pdf = orientation.uniform_pdf()
        
        n0=0
        # particle size distribution
        if psdargs is not None:
            scatterer.psd_integrator = PSDIntegrator(D_max = psdargs["D_max"], num_points=100)
            scatterer.psd = ExponentialPSD(N0=psdargs["N"], Lambda=psdargs["Lambda"])
    #         scatterer.psd = GammaPSD(D0 = psdargs["D_med"], Nw = psdargs["N"], mu=psdargs["mu"])
    
            # # Use only exponental at the moment. gamma not figured out yet
            n0 = self.numscatterers_pervolume(scatterer.psd, dist_type="Gamma", nf=psdargs["N"], Lambda=psdargs["Lambda"], mu=psdargs["mu"], D_max = psdargs["D_max"], D_med = psdargs["D_med"])
            
        return scatterer, n0
          
        
    def PhaseMatrix(self, scatterer, val, n0, theta_i, phi_i, theta_s, phi_s):
        
        """
        Calculates Phase matrix for randomly oriented spheroids averaged over orientation 
        (and size distribution hopefully) in the back scattering direction 
        
        Args:
        scatterer: instance of class pytmatrix.Scatterer 
        theta_i: incidence angle in radians
        phi_i: azimuth angle for incident direction in radians
        theta_s: backscatter angle in radians 
        phi_s: azimuth angle for incident direction in radians 
        
        Returns:
        A 4x4 phase matrix for the spheroidal inclusions/scatterers in the upper layer
        """

        geom = (np.rad2deg(theta_i), np.rad2deg(theta_s), np.rad2deg(phi_i), np.rad2deg(phi_s),
                np.rad2deg(scatterer.alpha), np.rad2deg(scatterer.beta))
        scatterer.set_geometry(geom)

        scatterer.psd_integrator.geometries = (geom,)
        scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
        
        P = n0 * scatterer.get_Z()
        return P

    def ExtinctionMatrixMish(self, scatterer, val, n0, theta, phi):
        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
                scatterer.alpha, scatterer.beta)
        scatterer.set_geometry(geom)

        scatterer.psd_integrator.geometries = (geom,)
        scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)

        SA_dyad = S = scatterer.get_S()          # # fvv and fhh appear equal?

        # # Tsang (1985) - eq. 5 in page 139 - attenuation rate matrix
        
        M = (1j * 2 * np.pi * n0 / self.k) * S
        
        
        # # Mischenko 2002 formula
        K11 = K22 = K33 = K44 = 2 * np.pi * n0 / self.k * (S[0,0] + S[1,1]).imag
        K12 = K21 = 2 * np.pi * n0 / self.k * (S[0,0] - S[1,1]).imag
        K13 = K31 =  -2 * np.pi * n0 / self.k * (S[0,1] + S[1,0]).imag
        K14 = K41 = 2 * np.pi * n0 / self.k * (-S[0,1] + S[1,0]).real
        K23 = 2 * np.pi * n0 / self.k * (-S[0,1] + S[1,0]).imag
        K32 = -K23
        K24 = -2 * np.pi * n0 / self.k * (S[0,1] + S[1,0]).real
        K42 = -K24
        K34 = 2 * np.pi * n0 / self.k * (S[1,1] - S[0,0]).real
        K43 = -K34
        
        K_e = np.matrix([[K11, K12, K13, K14], [K21, K22, K23, K24], [K31, K32, K33, K34], [K41, K42, K43, K44]])
        
# #         # # Tsang 1985 formula
#         K_e = np.matrix([[-2*M[0,0].real, 0, -M[0,1].real, -M[0,1].imag], \
#                          [0, -2*M[1,1].real, -M[1,0].real, M[1,0].imag], \
#                          [-2*M[1,0].real, -2*M[0,1].real, -M[0,0].real-M[1,1].real, M[0,0].imag-M[1,1].imag], \
#                          [2*M[1,0].imag, -2*M[0,1].imag, -M[0,0].imag+M[1,1].imag, -M[0,0].real-M[1,1].real]])
  
        beta, E = np.linalg.eig(K_e)
        Einv = np.linalg.inv(E)
        return beta, E, Einv
    
    def ExtinctionMatrix(self, scatterer, val, n0, theta, phi):
        
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

        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
                scatterer.alpha, scatterer.beta)
        scatterer.set_geometry(geom)

        scatterer.psd_integrator.geometries = (geom,)
        scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)

        SA_dyad = scatterer.get_S()          # # fvv and fhh appear equal?     

        # # Tsang (1985) - eq. 5 in page 139 - attenuation rate matrix
        
        M = (1j * 2 * np.pi * n0 / self.k) * SA_dyad
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
           
   
    def D(self, beta, theta, d, kab = 0):
        D = np.diag(np.exp((beta+kab) * - d/ np.cos(theta)))
        return D

            
    def ExtinctionCS(self, scatterer, n0, theta, phi, pol=None):
        
        if pol == None: pol = self.polH

        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
                np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))

        scatterer.set_geometry(geom)

        scatterer.psd_integrator.geometries = (geom,)
        scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
        
        SA_dyad = scatterer.get_S()

        # # use S_vv for vertical pol and S_hh for horizontal pol
        if pol == self.polV: 
            ext_cs = n0 * (4 * np.pi / self.k) * SA_dyad[0,0].imag
#             sca_cs = self.l1.inclusions.n0 * 4 * np.pi * (np.abs(SA_dyad[0,0]) ** 2 + np.abs(SA_dyad[0,1]) ** 2)
            
        elif pol == self.polH: 
            ext_cs = n0 * (4 * np.pi / self.k) * SA_dyad[1,1].imag
            
#             sca_cs = self.l1.inclusions.n0 * 4 * np.pi * (np.abs(SA_dyad[1,0]) ** 2 + np.abs(SA_dyad[1,1]) ** 2)
            
        # # make a diagonal matrix
        K_e = np.zeros((4,4))
        np.fill_diagonal(K_e, ext_cs)
        
        return ext_cs, K_e
    
    def ScatterCS(self, scatterer, theta, phi, pol):
        
        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
            np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))

        scatterer.set_geometry(geom,)
        scatterer.psd_integrator.geometries = (geom,)
        scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=True)
        ssa = scatter.ssa(scatterer, True)
        print("ssa = ", ssa)
        
        def ScatterInt(th, ph):

#             (scatterer.phi, scatterer.thet) = (np.rad2deg(ph), np.rad2deg(th))
            
            geom = (np.rad2deg(theta), np.rad2deg(th), np.rad2deg(phi), np.rad2deg(ph),
            np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))

            scatterer.set_geometry(geom,)
            scatterer.psd_integrator.geometries = (geom,)
            scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
            
            Z = scatterer.get_Z()
            if pol == self.polV:            
                scat_int = Z[0,0] + Z[0,1]
            elif pol == self.polH: 
                scat_int = Z[0,0] - Z[0,1]
                

            return scat_int * np.sin(theta)

        scat_cs, err = scipy.integrate.dblquad(ScatterInt, 0, 2*np.pi, lambda x: 0.0, lambda x: np.pi)        # first set of limits for phi and second set of limits for theta
        
        return scat_cs, err
    
    
    def integrate_withM(self, beta, theta, matrix, limits=[0.,0.], depth=0., kab=0.):
        n = len(beta)
        matrix_int = np.zeros_like(matrix)
        
        for i in range(n):
            for j in range(n):
                d = lambda z: np.exp(((beta[i]+kab) * (z + depth) / np.cos(theta))) * matrix[i,j]
                matrix_int[i,j], err = scipy.integrate.quad(d, limits[0], limits[1])
        return matrix_int
    
    def integrate(self, beta, theta, limits=[0.,0.], depth=0., kab=0.):
        n = len(beta)
        D = np.zeros((n,n))
        
        for i in range(n):
            for j in range(n):
                if i == j: 
                    d = lambda z: np.exp(((beta[i]+kab) * (z + depth) / np.cos(theta))) 
                    D[i,j], err = scipy.integrate.quad(d, limits[0], limits[1])
        return D    
    
    def Mueller_bed(self, val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12, scat, n0=0):
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
        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrixMish(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrixMish(scat, val, n0, thetat_rad, self.phi_s)
        M_bed = np.linalg.multi_dot([T10_coh, E_plus, self.D(beta_plus, thetat_rad, val["d"], k_a_medium), Einv_plus, R12, E_minus, self.D(beta_minus, thetat_rad, val["d"], k_a_medium), Einv_minus, T01_coh])
        
        return M_bed
                
    
    def Mueller_bedvol(self, val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12_coh, scat, n0):
        
        """
        Compute the real-valued 4x4 Mueller matrix for scattering from the 
        inclusions in the first layer (2) and first layer - substrate interface (1).
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """
        
        # # Extinction due to scatterers
        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrixMish(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_plus1, E_plus1, Einv_plus1 = self.ExtinctionMatrixMish(scat, val, n0, thetat_rad, self.phi_i)
        beta_plus2, E_plus2, Einv_plus2 = self.ExtinctionMatrixMish(scat, val, n0, thetat_rad, self.phi_s)
        
        Dminus = self.D(beta_minus, thetat_rad, -self.l1.d, kab = k_a_medium)
        Dplus1 = self.integrate(beta_plus1, thetat_rad, [-val["d"], 0], kab = k_a_medium, depth = -self.l1.d)
        Dplus2 = self.integrate(beta_plus2, thetat_rad, [-val["d"], 0], kab = k_a_medium)
        
        EDEminus = np.linalg.multi_dot([E_minus, Dminus, Einv_minus])
        EDEplus1 = np.linalg.multi_dot([E_plus1, Dplus1, Einv_plus1])
        EDEplus2 = np.linalg.multi_dot([E_plus2, Dplus2, Einv_plus2])

        # # Phase matrix for scatterers
        P = self.PhaseMatrix(scat, val, n0, np.pi/2 - thetat_rad, self.phi_i, np.pi/2 - thetat_rad, self.phi_s)
        
        M_bedvol =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), EDEplus2, P, EDEplus1, R12_coh, EDEminus, T01_coh])
               
        return M_bedvol
    
    def Mueller_volbed(self, val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12_coh, scat, n0):

        """
        Compute the real-valued 4x4 Mueller matrix for scattering from the 
        inclusions in the first layer (1) and first layer - substrate interface (2).
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """
        # # Extinction due to scatterers
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrixMish(scat, val, n0, thetat_rad, self.phi_s)
        beta_minus1, E_minus1, Einv_minus1 = self.ExtinctionMatrixMish(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_minus2, E_minus2, Einv_minus2 = self.ExtinctionMatrixMish(scat, val, n0, np.pi - thetat_rad, self.phi_s)
        
        Dminus1 = self.integrate(beta_minus1, thetat_rad, [-val["d"], 0], kab = k_a_medium)
        Dminus2 = self.integrate(beta_minus2, thetat_rad, [-val["d"], 0], kab = k_a_medium, depth = -self.l1.d)
        Dplus = self.D(beta_plus, thetat_rad, -self.l1.d, kab = k_a_medium)
        
        EDEminus1 = np.linalg.multi_dot([E_minus1, Dminus1, Einv_minus1])
        EDEminus2 = np.linalg.multi_dot([E_minus2, Dminus2, Einv_minus2])
        EDEplus = np.linalg.multi_dot([E_plus, Dplus, Einv_plus])

        # # Phase matrix for scatterers
        P = self.PhaseMatrix(scat, val, n0, np.pi - (np.pi/2 - thetat_rad), self.phi_i, np.pi - (np.pi/2 - thetat_rad), self.phi_s)
       
        M_volbed =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), EDEplus, R12_coh, EDEminus2, P, EDEminus1, T01_coh])
        
        return M_volbed
    
    def Mueller_vol(self, val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, scat, n0=0):  
        
        """
        Compute the real-valued 4x4 Mueller matrix for scattering 
        from the inclusions in the first layer.
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        poltype: polarization type "lin" for linear or "circ" for circular
        
        Returns:
        M_bed: A 4x4 real-valued Mueller matrix
        """
        # # Extinction due to scatterers
        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrixMish(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrixMish(scat, val, n0, thetat_rad, self.phi_s)
#         if val["abyc"] == 1:
#             k_eminus, K_Eminus = self.ExtinctionCS(scat, n0, np.pi - thetat_rad, self.phi_i)
#             k_eplus, K_Eplus = self.ExtinctionCS(scat, n0, thetat_rad, self.phi_s)
#             EDEminus = self.integrate(np.diagonal(K_Eminus), thetat_rad, [-val["d"], 0], k_a_medium)
#             EDEplus = self.integrate(np.diagonal(K_Eplus), thetat_rad, [-val["d"], 0], k_a_medium)
            
#         else:
        Dminus = self.integrate(beta_minus, thetat_rad, [-val["d"], 0], kab = k_a_medium)
        Dplus = self.integrate(beta_plus, thetat_rad, [-val["d"], 0], kab = k_a_medium)
        EDEminus = np.linalg.multi_dot([E_minus, Dminus, Einv_minus])
        EDEplus = np.linalg.multi_dot([E_plus, Dplus, Einv_plus])

        # # Phase matrix for scatterers
        P = self.PhaseMatrix(scat, val, n0, np.pi - thetat_rad, self.phi_i, thetat_rad, self.phi_s)
        
        M_vol =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), EDEplus, P, EDEminus,T01_coh])
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
    
    def surfaceBSC(self, val, crosspol="False"):
        self.setGeometricOptics()
        self.setLayerProperties()
        
        thetai_rad =  np.deg2rad(val["thetai"])    
        
        try:
            thi =  val["thetai"].item()
        except:
            thi = val["thetai"]
        try:
            ks = val["ks1"].item()
        except:
            ks = val["ks1"]
        try:
            eps = val["eps1"].item()/self.atm.eps
        except:
            eps = val["eps1"]/self.atm.eps

        
        i01 = RoughSurface(self.wavelength, self.atm.eps, val["eps1"], val["ks1"], self.wavelength, autocorr_fn = "exponential", theta_i = thetai_rad, theta_s = thetai_rad)
        if crosspol=="True":
            eng = matlab.engine.start_matlab()
            svv0, shh0, svh0 = eng.I2EM_Backscatter_model(self.nu/1e9, ks, self.wavelength, thi, eps, 1, 0, nargout=3)
            cpr = self.sigma2cpr(svv0, shh0, svh0)
        else:
            svv, shh = i01.Copol_BSC()
            svv0, shh0 = self.sigma2sigma0([svv, shh])
            cpr = 0.0
        
        return svv0, shh0, cpr
    
    
    def subsurfaceBSC(self, val, crosspol="False"):
        self.setGeometricOptics()
        self.setLayerProperties()
        
        thetai_rad =  np.deg2rad(val["thetai"])  
        thetat_rad = self.TransmissionAngle(thetai_rad, val["eps1"], val["eps2"])
        
        thi = np.rad2deg(thetat_rad).item()
        try:
            ks = val["ks2"].item()
        except:
            ks = val["ks2"]
        try:
            eps = (val["eps2"]/val["eps1"]).item()
        except:
            eps = val["eps2"]/val["eps1"]
     
        # # Reflection from subsurface interface
        if crosspol=="True":
                eng = matlab.engine.start_matlab()
                svv0, shh0, svh0 = eng.I2EM_Backscatter_model(self.nu/1e9, ks, self.wavelength, thi, eps, 1, 0, nargout=3)
                svv, shh, svh = self.sigma02sigma([svv0, shh0, svh0])     
        else:
            i12 = RoughSurface(self.wavelength, val["eps1"], val["eps2"], val["ks2"], self.wavelength, autocorr_fn = "exponential", theta_i = thetat_rad, theta_s = thetat_rad)
            svv, shh = i12.Copol_BSC()
            svh = 0.0
        R12nc = self.sigma2Mueller(svv, shh, svh)
        
        
        # # Transmission through in to the upper layer
        T01c = self.CoherentTransMatrix(self.atm.eps, val["eps1"], val["ks1"], thetai_rad)
        T10c = self.CoherentTransMatrix(val["eps1"], self.atm.eps, val["ks1"], thetat_rad)
        
        # # Background absorption ulaby equation 11.47
        k_a_medium = 2 * self.k * np.sqrt(val["eps1"]).imag
        
        
        # # subsurface scattering with buried scatterers
        if val["a"] != 0 and val["epsinc"].real !=0:
            print("scatterers present")
            # # Create a T-matrix instance for volume scattering calculations (phase matrix and extinction matrix)
            D_med = val["a"]          # median diameter in m
            D_max = 2 * val["a"]       # maximum diameter in m
            Lambda = 1          # some parameter to change for particle size?
            mu = 1
            N = 5000 
            tm, n0 = self.Tmatrixcalc(val["a"], val["epsinc"].real ** 0.5, val["abyc"], val["alpha"], val["beta"], D_med=D_med, D_max=D_max, Lambda=Lambda, mu=mu, N=N)
            Msub = self.Mueller_bed(val, thetai_rad, thetat_rad, k_a_medium, T01c, T10c, R12nc, tm, n0)

        # # subsurface scattering without buried scatterers
        else:                
            Msub = T01c * np.exp(k_a_medium * -val["d"] /np.cos(thetat_rad)) * R12nc * np.exp(k_a_medium * -val["d"] /np.cos(thetat_rad)) * T10c
            
        svv, shh = self.Mueller2sigma(Msub)    
        svv0, shh0 = self.sigma2sigma0([svv, shh]) 
        cpr = self.Mueller2cpr(Msub)
        
        return svv0, shh0, cpr
    
    
    def volumeBSC(self, val, crosspol="False"):
        self.setGeometricOptics()
        self.setLayerProperties()
        
        thetai_rad = np.deg2rad(val["thetai"])
        thetat_rad = self.TransmissionAngle(thetai_rad, self.atm.eps, val["eps1"])
        
        # # transmissivity into layer
        T01_coh = self.CoherentTransMatrix(self.atm.eps, val["eps1"], val["ks1"], thetai_rad)
        T10_coh = self.CoherentTransMatrix(val["eps1"], self.atm.eps, val["ks1"], thetat_rad)
        
        # # Extinction due to uniform background medium ulaby equation 11.47
        k_a_medium = 2 * self.k * np.sqrt(val["eps1"]).imag

        # # Create a T-matrix instance for volume scattering calculations (phase matrix and extinction matrix)
        if val["a"] != 0 and val["epsinc"].real !=0:
            D_med = val["a"]          # median diameter in m
            D_max = 2 * val["a"]       # maximum diameter in m
            Lambda = 1          # some parameter to change for particle size?
            mu = 1
            N = 1000         # increase for increasing particle concentration (n0); don't go over N=10000 0r 15000
            # # parameters for particle size distribution

            tm, n0 = self.Tmatrixcalc(val["a"], val["epsinc"].real ** 0.5, val["abyc"], val["alpha"], val["beta"], D_med=D_med, D_max=D_max, Lambda=Lambda, mu=mu, N=N)

        Mvol = self.Mueller_vol(val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, tm, n0)
        svv, shh = self.Mueller2sigma(Mvol)
        svv, shh = self.sigma2sigma0([svv, shh])
        cpr = self.Mueller2cpr(Mvol)
        
        return svv, shh, cpr
    
    def volumesubBSC(self, val, crosspol="False"):
        
        self.setGeometricOptics()
        self.setLayerProperties()
        
        thetai_rad = np.deg2rad(val["thetai"])
        thetat_rad = self.TransmissionAngle(thetai_rad, self.atm.eps, val["eps1"])
        
        # # transmissivity into layer
        T01_coh = self.CoherentTransMatrix(self.atm.eps, val["eps1"], val["ks1"], thetai_rad)
        T10_coh = self.CoherentTransMatrix(val["eps1"], self.atm.eps, val["ks1"], thetat_rad)
        
        # # Extinction due to uniform background medium ulaby equation 11.47
        k_a_medium = 2 * self.k * np.sqrt(val["eps1"]).imag
        
        # # Reflection from subsurface interface
        R12_coh = self.CoherentRefMatrix(val["eps1"], val["eps2"], val["ks2"], thetat_rad)

        # # Create a T-matrix instance for volume scattering calculations (phase matrix and extinction matrix)
        if val["a"] != 0 and val["epsinc"].real !=0:
            D_med = val["a"]          # median diameter in m
            D_max = 2 * val["a"]       # maximum diameter in m
            Lambda = 1          # some parameter to change for particle size?
            mu = 1
            N = 1000         # increase for increasing particle concentration (n0); don't go over N=10000 0r 15000
            # # parameters for particle size distribution

            tm, n0 = self.Tmatrixcalc(val["a"], val["epsinc"].real ** 0.5, val["abyc"], val["alpha"], val["beta"], D_med=D_med, D_max=D_max, Lambda=Lambda, mu=mu, N=N)

        Mvolsub = self.Mueller_volbed(val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12_coh, tm, n0) + self.Mueller_bedvol(val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12_coh, tm, n0)
        svv, shh = self.Mueller2sigma(Mvolsub)
        svv, shh = self.sigma2sigma0([svv, shh])
        cpr = self.Mueller2cpr(Mvolsub)
        
        return svv, shh, cpr

    
    def VRTsolver(self, val_dict, scattertype):
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
        
        # # output placeholders
        
        shh_sur = svv_sur = cpr_sur = shh_sub = svv_sub = cpr_sub = shh_vol = svv_vol = cpr_vol = shh_volsub = svv_volsub = cpr_volsub = np.nan
               
        if scattertype == "surface":
            svv_sur, shh_sur, cpr_sur = self.surfaceBSC(val_dict, "False")
            try:
                svv_sur = svv_sur.real
                shh_sur = shh_sur.real
            except:
                pass
               
        if scattertype == "subsurface":
            svv_sub, shh_sub, cpr_sub = self.subsurfaceBSC(val_dict, "True")
            try:
                svv_sub = svv_sub.real
                shh_sub = shh_sub.real
            except:
                pass
               
        if scattertype == "volume":
            svv_vol, shh_vol, cpr_vol = self.volumeBSC(val_dict)
            try:
                svv_vol = svv_vol.real
                shh_vol = shh_vol.real
            except:
                pass
            
        if scattertype == "volume-subsurface":
            svv_volsub, shh_volsub, cpr_volsub = self.volumesubBSC(val_dict)
            try:
                svv_volsub = svv_vol.real
                shh_volsub = shh_vol.real
            except:
                pass
               
        # # make a pandas series
        cols = list(val_dict.keys())
        output_cols = ["shh_sur", "svv_sur", "cpr_sur", "shh_sub", "svv_sub", "cpr_sub", "shh_vol", "svv_vol", "cpr_vol", "shh_volsub", "svv_volsub", "cpr_volsub"]
        cols.extend(output_cols) 
               
        values = list(val_dict.values())
        output_values = [shh_sur, svv_sur, cpr_sur, shh_sub, svv_sub, cpr_sub, shh_vol, svv_vol, cpr_vol, shh_volsub, svv_volsub, cpr_volsub]
        values.extend(output_values)

        return values
            
               
        # # check if this is necessary

#         self.R01 = self.NonCoherentRefMatrix(self.i01, self.l1.theta_i)
#         self.R12 = self.NonCoherentRefMatrix(self.i12, self.l1.theta_t)
        
#         self.R01_coh = self.CoherentRefMatrix(self.atm, self.l1, self.l1.theta_i)
#         self.R12_coh = self.CoherentRefMatrix(self.l1, self.l2, self.l1.theta_t)
                                  
#         # # adding a factor in front of transmission matrix based on equation 11.61b in Ulaby big book
                                          
#         self.T01 = self.NonCoherentTransMatrix(self.i01, self.l1.theta_i)
#         self.T10 = self.NonCoherentTransMatrix(self.i10, self.l2.theta_s)
        
#         self.T01_coh = self.CoherentTransMatrix(self.atm, self.l1, self.l1.theta_i)
#         self.T10_coh = self.CoherentTransMatrix(self.l1, self.atm, self.l2.theta_s)

#         # # Create a T-matrix instance for phase matrix and extinction matrix calculations
#         tm = self.Tmatrixcalc(self.l1)
  
#         # # VRT
        
#         sigma_hh, DLP = self.Mueller_lin(tm, self.polH)
#         sigma_vv, _ = self.Mueller_lin(tm, self.polV)
        
# #         cpr = self.Mueller_circ(tm, "LC")
#         cpr = np.zeros(5)

        
#         sigma0_hh, sigma0_vv = self.sigma2sigma0([sigma_hh, sigma_vv])

#         e_v, e_h = self.RT0_emission(tm)
        
#         return [sigma0_vv, sigma0_hh, cpr, e_v, e_h]
        
        # # only for plotting fresnel coefficients
#         refh,transh,rh,th = self.FresnelCoefH(self.atm, self.l1, self.theta_i)
#         refv,transv,rv,tv = self.FresnelCoefV(self.atm, self.l1, self.theta_i)
#         return [sigma0_vv, sigma0_hh, CPR, refh, refv, transh, transv, rh, rv, th, tv]
        
#         # # only for plotting surface backscatter
#         svv1, svv2, shh1, shh2, stvv1, sthh1, stvv2, sthh2 = self.reftranscomp()
#         return svv1, svv2, shh1, shh2, stvv1, sthh1, stvv2, sthh2


        # surface backscatter
#         svv0, sigmavv0, shh0, sigmahh0 = self.I2EM(self.atm, self.l1)
        
#         eng = matlab.engine.start_matlab()
#         svv0, shh0, sigma0hv = eng.I2EM_Backscatter_model(self.nu/1e9, self.l1.upperks, self.l1.corrlen, self.theta_i.item(), self.l1.eps/self.atm.eps, 1, 0, nargout=3)
#         svv, shh = self.i01.Copol_BSC()
#         return svv, shh
#         sigmavv0, sigmahh0 = self.sigma2sigma0([shh, svv])
#         return svv0, sigmavv0, shh0, sigmahh0

    def RT0_emission(self, scatterer):
        """
        Primary function that executes the oth order VRT emission model
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        
        Returns:
        e_v: Emissivity in V polarization
        e_h: Emissivity in H polarization
        """
        
        self.setGeometricOptics()
        self.setLayerProperties()
        
        beta, E, Einv = self.ExtinctionMatrix(scatterer, self.l2.theta_s, self.phi_s)
        trans = np.linalg.multi_dot([E, self.D(beta, self.l2.theta_s, self.l1.d), Einv])


        # # resetting scatterer geometry after extinction matrix calculation
        geom = (np.rad2deg(self.theta_s), np.rad2deg(self.theta_s), np.rad2deg(self.phi_s), np.rad2deg(self.phi_s),
                np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))
        scatterer.set_geometry(geom,)
        scatterer.psd_integrator.geometries = (geom,)
        scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=True)
#         ke_v, _ = self.ExtinctionCS(scat, self.theta_s, self.phi_s, self.polV)
#         ke_h, _ = self.ExtinctionCS(scat, self.theta_s, self.phi_s, self.polH)
        
#         ks_v, _ = self.ScatterCS(scat, self.theta_s, self.phi_s, self.polV)
#         ks_h, _ = self.ScatterCS(scat, self.theta_s, self.phi_s, self.polH)
        
        ke_v = scatter.ext_xsect(scatterer, h_pol=False)
        ke_h = scatter.ext_xsect(scatterer, h_pol=True)
        
        ks_v = scatter.sca_xsect(scatterer, h_pol=False)
        ks_h = scatter.sca_xsect(scatterer, h_pol=True)
        
        ssa_v = scatter.ssa(scatterer, h_pol=False)
        ssa_h = scatter.ssa(scatterer, h_pol=True)
                
#         ssa_h = ks_h/ke_h
#         ssa_v = ks_v/ke_v

        print(self.l1.eps)
        print("SSA = ", ssa_h, ssa_v)
        
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

       
