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

from pytmatrix import tmatrix, orientation, scatter, refractive
from pytmatrix.psd import PSDIntegrator, GammaPSD, ExponentialPSD, UnnormalizedGammaPSD

# import tmatrix, orientation, scatter
# from psd import PSDIntegrator, GammaPSD, ExponentialPSD

import radar_inst as radar
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
    
    
    def __init__(self):

        # Constants
        self.wavelength = radar.wavelength                     # wavelength in meters
        self.nu = c / self.wavelength
        self.k = 2 * np.pi / self.wavelength
        
         ## Magellan altitude at periapsis
        self.R = radar.altitude                   # 250 km periapsis; 8029 km apoapsis
        
        # Polarization information
        self.polV = radar.polarization1
        self.polH = radar.polarization2           
    
    def setGeometricOptics(self):
        
        """
        Sets the viewing geometry properties (mainly angles) for each layer.
        """
        self.phi_i = np.deg2rad(0)
        self.phi_s = np.pi - self.phi_i

        
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
    
    def Muellermod2sigma(self, M):
        svv = 4 * np.pi * M[0,0]
        shh = 4 * np.pi * M[1,1]
        return svv, shh
        
    def Mueller2sigma(self, M):
        # # multiply by 4*pi*cos theta if needed
        # # division by a factor of 4 or 2?
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
        m33 = 2* (s11*s22)**0.5
        m34 = 0
#         m44 = 2*shv
        m44 = 2* (s11*s22)**0.5
        
        M = np.array([[m11,m12,m13,m14], [m12,m22,m23,m24], [m13, m23, m33, m34], [-m14, -m24, -m34, m44]]) / 4
        # Modified Mueller materix form
        B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) 
        M = B*M*np.linalg.inv(B)
        
        return M
       
    
    def Mueller2cpr(self,M, poltype="linear"):
        
        # # cpr = S1-S4 / S1+S4 = Iv+Ih-V / Iv+Ih+V

        if poltype == "linear":
            Is = np.linalg.multi_dot([M, np.array([0,1,0,0])])
        elif poltype == "circular":
            Is = np.linalg.multi_dot([M, np.array([0.5,0.5,0,1])])
        
        cpr = (Is[0,0] + Is[0,1] - Is[0,3]) / (Is[0,0] + Is[0,1] + Is[0,3])
        
        return cpr
    
    def Mueller2dlp(self,M, poltype="linear"):
        
        # # dlp = np.sqrt(S2**2 + S3**2) / S1 = Iv+Ih-V / Iv+Ih+V
    
        if poltype == "linear":
            Is = np.linalg.multi_dot([M, np.array([0,1,0,0])])
        elif poltype == "circular":
            Is = np.linalg.multi_dot([M, np.array([0.5,0.5,0,1])])
        
        dlp =  np.sqrt((Is[0,0] - Is[0,1])**2 + Is[0,2]**2) / (Is[0,0] + Is[0,1])
        
        return dlp
    
    def sigma2dlp(self, svv, shh, svh=0, poltype = "circular"):
        
        M = self.sigma2Mueller(svv, shh, svh)
        return self.Mueller2dlp(M, poltype)
    
    def sigma2cpr(self, svv, shh, svh):
        cpr = (svv+shh+2*np.sqrt(svv*shh))/(svv+shh-2*np.sqrt(svv*shh))       
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

        # # Mueller matrix convention
        T = 0.5 * np.matrix([[tv+th, tv-th, 0., 0.], [tv-th, tv+th, 0., 0.], [0., 0., 2*(tv*th)**0.5, 0.], [0., 0., 0., 2*(tv*th)**0.5]])
        
        
        # Modified Mueller materix form
        B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) 
        T = B*T*np.linalg.inv(B)
        
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
        
        # Modified Mueller materix form
        B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) 
        R = B*R*np.linalg.inv(B)

        R_coh = loss_factor * R

        return R_coh
    

    def numscatterers_pervolume(self, psd, dist_type="Gamma", Nw=1, Lambda=1, mu=100, D_max = 0.06, D_med = 1):
        
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
        
        if dist_type == "gamma":
            nf = Nw * 6.0/3.67**4 * (3.67+mu)**(mu+4)/gamma(mu+4)
            Dfunc = lambda d: nf * np.exp(mu*np.log(d)-(3.67+mu)*(d))
            D_func_int, err = scipy.integrate.quad(Dfunc, 0.0001, np.inf)
            
        elif dist_type == "exponential":
            Dfunc = lambda d: Nw * np.exp(-Lambda*d)
            D_func_int, err = scipy.integrate.quad(Dfunc, 0.0001, D_max)    # # using 0.0001 as lower limit to avoid division by 0
            
        elif dist_type == "ungamma":
            Dfunc = lambda d: Nw * np.exp(mu*np.log(d)-Lambda*d)
            D_func_int, err = scipy.integrate.quad(Dfunc, 0.0001, np.inf)    # # using 0.0001 as lower limit to avoid division by 0
            
        return D_func_int
       
        
    def Tmatrixcalc(self, wavelength, radius, rindex_tup, volfrac_tup, axis_ratio, alpha, beta, **psdargs):

        """
        Initializes a scatterer object of class pytmatrix.Scatterer
        Make sure particle size and wavelength have the same unit
        Current units: meters
        
        Args:
        l1: upper layer (instance of class Layers) containing the scatterers
        
        Returs:
        Instance of class pytmatrix.Scatterer initialized woth values from vrt.l1 
        
        """

        ri = refractive.mg_refractive(rindex_tup, volfrac_tup)
        scatterer = tmatrix.Scatterer(wavelength = wavelength, 
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
            
            if psdargs["psdfunc"] == "exponential":
                scatterer.psd = ExponentialPSD(N0=psdargs["N"], Lambda=psdargs["Lambda"], D_max = psdargs["D_max"])            
            elif psdargs["psdfunc"] == "gamma":
                scatterer.psd = GammaPSD(D0 = psdargs["D_max"]/2, Nw = psdargs["N"], mu=psdargs["mu"])
            elif psdargs["psdfunc"] == "ungamma":
                scatterer.psd = UnnormalizedGammaPSD(N0 = psdargs["N"], Lambda=psdargs["Lambda"], mu=psdargs["mu"], D_max = psdargs["D_max"])

            n0 = self.numscatterers_pervolume(scatterer.psd, dist_type=psdargs["psdfunc"], Nw=psdargs["N"], Lambda=psdargs["Lambda"], mu=psdargs["mu"], D_max = psdargs["D_max"], D_med = psdargs["D_med"])
            
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
                val["alpha"], val["beta"])
        
        scatterer.set_geometry(geom)

        if scatterer.psd_integrator != None:
            scatterer.psd_integrator.geometries = (geom,)
            scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
        
        P = n0 * scatterer.get_Z()
        
        # Modified Mueller materix form
        B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) 
        P = B*P*np.linalg.inv(B)

        return P

    def ExtinctionMatrixMish(self, scatterer, val, n0, theta, phi):
        geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
               val["alpha"], val["beta"])
        scatterer.set_geometry(geom)

        if scatterer.psd_integrator != None:
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
        B = np.matrix([[0.5, 0.5, 0., 0.], [0.5, -0.5, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) 
        K_e = B*K_e*np.linalg.inv(B)
  
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
                val["alpha"], val["beta"])
        scatterer.set_geometry(geom)

        if scatterer.psd_integrator != None:
            scatterer.psd_integrator.geometries = (geom,)
            scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)

        SA_dyad = scatterer.get_S()          # # fvv and fhh appear equal? 
        # # Tsang (1985) - eq. 5 in page 139 - attenuation rate matrix
        
        M = (1j * 2 * np.pi * n0 / self.k) * SA_dyad
#         M = n0 * SA_dyad
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

            
    def ExtinctionCS(self, scatterer, val, n0, theta, phi, pol=None):
        
        if pol == None: pol = self.polH

#         geom = (np.rad2deg(theta), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(phi),
#                 val["alpha"], val["beta"])

#         scatterer.set_geometry(geom)
#         if scatterer.psd_integrator != None:
#             scatterer.psd_integrator.geometries = (geom,)
#             scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=False)
        
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
        
        if scatterer.psd_integrator != None:
            scatterer.psd_integrator.geometries = (geom,)
            scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=True)
        ssa = scatter.ssa(scatterer, True)
        
        def ScatterInt(th, ph):

#             (scatterer.phi, scatterer.thet) = (np.rad2deg(ph), np.rad2deg(th))
            
            geom = (np.rad2deg(theta), np.rad2deg(th), np.rad2deg(phi), np.rad2deg(ph),
            np.rad2deg(self.l1.inclusions.alpha), np.rad2deg(self.l1.inclusions.beta))

            scatterer.set_geometry(geom,)
            if scatterer.psd_integrator != None:
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
        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrix(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrix(scat, val, n0, thetat_rad, self.phi_s)
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
        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrix(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_plus1, E_plus1, Einv_plus1 = self.ExtinctionMatrix(scat, val, n0, thetat_rad, self.phi_i)
        beta_plus2, E_plus2, Einv_plus2 = self.ExtinctionMatrix(scat, val, n0, thetat_rad, self.phi_s)
 
        
        Dminus = self.D(beta_minus, thetat_rad, val["d"], kab = k_a_medium)
        Dplus1 = self.integrate(beta_plus1, thetat_rad, [-val["d"], 0], kab = k_a_medium, depth = -val["d"])
        Dplus2 = self.integrate(beta_plus2, thetat_rad, [-val["d"], 0], kab = k_a_medium)
        
        EDEminus = np.linalg.multi_dot([E_minus, Dminus, Einv_minus])
        EDEplus1 = np.linalg.multi_dot([E_plus1, Dplus1, Einv_plus1])
        EDEplus2 = np.linalg.multi_dot([E_plus2, Dplus2, Einv_plus2])

        # # Phase matrix for scatterers
        P = self.PhaseMatrix(scat, val, n0, np.pi/2 - thetat_rad, self.phi_i, np.pi/2 - thetat_rad, self.phi_s)
        
        M_bedvol =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), EDEplus2, P, EDEplus1, R12_coh, EDEminus, T01_coh])
        
        # Tsang formulation
#         ext_term = (np.exp(-beta_plus2*val["d"]/np.cos(thetat_rad)) + np.exp(-beta_plus1*val["d"]/np.cos(thetat_rad))) \
#                     / ((beta_plus1/np.cos(thetat_rad)) + (beta_plus2/np.cos(thetat_rad)))
        
#         M_bedvol =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), E_plus2, Einv_plus2, P, E_plus1, np.diag(ext_term), Einv_plus1, R12_coh, EDEminus, T01_coh])
               
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
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrix(scat, val, n0, thetat_rad, self.phi_s)
        beta_minus1, E_minus1, Einv_minus1 = self.ExtinctionMatrix(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_minus2, E_minus2, Einv_minus2 = self.ExtinctionMatrix(scat, val, n0, np.pi - thetat_rad, self.phi_s)

        
        Dminus1 = self.integrate(beta_minus1, thetat_rad, [-val["d"], 0], kab = k_a_medium)
        Dminus2 = self.integrate(beta_minus2, thetat_rad, [-val["d"], 0], kab = k_a_medium, depth = -val["d"])
        Dplus = self.D(beta_plus, thetat_rad, val["d"], kab = k_a_medium)
        
        EDEminus1 = np.linalg.multi_dot([E_minus1, Dminus1, Einv_minus1])
        EDEminus2 = np.linalg.multi_dot([E_minus2, Dminus2, Einv_minus2])
        EDEplus = np.linalg.multi_dot([E_plus, Dplus, Einv_plus])

        # # Phase matrix for scatterers
        P = self.PhaseMatrix(scat, val, n0, np.pi - (np.pi/2 - thetat_rad), self.phi_i, np.pi - (np.pi/2 - thetat_rad), self.phi_s)
       
        M_volbed =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), EDEplus, R12_coh, EDEminus2, P, EDEminus1, T01_coh])
        
        # # Tsang formulation
#         ext_term = (np.exp(-beta_minus2*val["d"]/np.cos(thetat_rad)) + np.exp(-beta_minus1*val["d"]/np.cos(thetat_rad))) \
#                     / ((beta_minus1/np.cos(thetat_rad)) + (beta_minus2/np.cos(thetat_rad)))
        
#         M_volbed =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), EDEplus, R12_coh, E_minus2, Einv_minus2, P, E_minus1, np.diag(ext_term), Einv_minus1, T01_coh])
        
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
        beta_minus, E_minus, Einv_minus = self.ExtinctionMatrix(scat, val, n0, np.pi - thetat_rad, self.phi_i)
        beta_plus, E_plus, Einv_plus = self.ExtinctionMatrix(scat, val, n0, thetat_rad, self.phi_s)

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
        
        # # Tsang formulation
        ext_term = (1 - np.exp(-(beta_plus*val["d"]/np.cos(thetat_rad))-(beta_minus*val["d"]/np.cos(thetat_rad))))/((beta_plus/np.cos(thetat_rad)) + (beta_minus/np.cos(thetat_rad)))
        M_vol =  np.linalg.multi_dot([T10_coh / np.cos(thetai_rad), E_plus,Einv_plus,P,E_minus,np.diag(ext_term),Einv_minus,T01_coh])
            
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

    def I2EM_emissivity(self, thetai, eps1, eps2, s, cl):
        
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
        
        try:
            thetai =  thetai.item()
        except:
            pass
        try:
            s = s.item()
        except:
            pass
        try:
            cl = cl.item()
        except:
            pass
        try:
            eps = (eps2/eps1).item()
        except:
            eps = eps2/eps1
        
        # # if float is not iterable error rises, check if the matlab function returns are separated by commas
        eng = matlab.engine.start_matlab()
        e_v, e_h = eng.I2EM_Emissivity_model(self.nu/1e9, s, cl, thetai, eps, 1, nargout=2)           # using .item() to convert from numpy float64 to python scalar

        return e_v, e_h
    
    def surfaceBSC(self, val, crosspol=False, emission = False):
        
        self.setGeometricOptics()

        
        thetai_rad =  np.deg2rad(val["thetai"])    
        
        try:
            thi =  val["thetai"].item()
        except:
            thi = val["thetai"]
        try:
            s1 = val["s1"].item()
        except:
            s1 = val["s1"]
        try:
            cl1 = val["cl1"].item()
        except:
            cl1 = val["cl1"]
        try:
            eps = (complex(val["eps1r"],val["eps1i"]) /val["atm_eps"]).item()
        except:
            eps = complex(val["eps1r"],val["eps1i"])/val["atm_eps"]

        i01 = RoughSurface(self.wavelength, val["atm_eps"], complex(val["eps1r"],val["eps1i"]), val["s1"], val["cl1"], autocorr_fn = "exponential", theta_i = thetai_rad, theta_s = thetai_rad)
        if crosspol==True:
            eng = matlab.engine.start_matlab()
            svv0, shh0, svh0 = eng.I2EM_Backscatter_model(self.nu/1e9, s1, cl1, thi, eps, 1, 0, nargout=3)
            svv, shh, svh = self.sigma02sigma([svv0, shh0, svh0])
            # # to ue sigma or sigma0?
            cpr = self.sigma2cpr(svv0, shh0, svh0)
            dlp = self.sigma2dlp(svv0, shh0, svh0)
        else:
            svv, shh = i01.Copol_BSC()
            svv0, shh0 = self.sigma2sigma0([svv, shh])
            cpr = 0.0
            dlp = 0.0
            
        if emission == True:
            ev, eh, _, _ = self.RT0_emission("surface", val, thetai_rad, thetat_rad = 0, k_a_medium = 0, scatterer = None,  n0 = 0)
        else:
            ev = eh = 0
        
        return svv, svv0, shh, shh0, cpr, dlp, ev, eh
    
    
    def subsurfaceBSC(self, val, crosspol=True, emission = True):
        self.setGeometricOptics()
        
        thetai_rad =  np.deg2rad(val["thetai"])  
        thetat_rad = self.TransmissionAngle(thetai_rad, val["atm_eps"], complex(val["eps1r"],val["eps1i"]))
        
        thi = np.rad2deg(thetat_rad).item()
        try:
            s2 = val["s2"].item()
        except:
            s2 = val["s2"]
        try:
            cl2 = val["cl2"].item()
        except:
            cl2 = val["cl2"]
        try:
            eps = (complex(val["eps2r"],val["eps2i"])/complex(val["eps1r"],val["eps1i"])).item()
        except:
            eps = complex(val["eps2r"],val["eps2i"])/complex(val["eps1r"],val["eps1i"])
     
        # # Reflection from subsurface interface
        if crosspol==True:
                eng = matlab.engine.start_matlab()
                svv0, shh0, svh0 = eng.I2EM_Backscatter_model(self.nu/1e9, s2, cl2, thi, eps, 1, 0, nargout=3)
                svv, shh, svh = self.sigma02sigma([svv0, shh0, svh0])     
        else:
            i12 = RoughSurface(self.wavelength, complex(val["eps1r"],val["eps1i"]), complex(val["eps2r"],val["eps2i"]), val["s2"], val["cl2"], autocorr_fn = "exponential", theta_i = thetat_rad, theta_s = thetat_rad)
            svv, shh = i12.Copol_BSC()
            svh = 0.0
            svv0, shh0 = self.sigma2sigma0([svv, shh])
            
        R12nc = self.sigma2Mueller(svv, shh, svh)

        # # Transmission through in to the upper layer
        T01c = self.CoherentTransMatrix(val["atm_eps"], complex(val["eps1r"],val["eps1i"]), val["s1"], thetai_rad)
        T10c = self.CoherentTransMatrix(complex(val["eps1r"],val["eps1i"]), val["atm_eps"], val["s1"], thetat_rad)
        
        # # Background absorption ulaby equation 11.47
        k_a_medium = 2 * self.k * np.sqrt(val["eps1i"])
        Msub = T01c * np.exp(k_a_medium * -val["d"] /np.cos(thetat_rad)) * R12nc * np.exp(k_a_medium * -val["d"] /np.cos(thetat_rad)) * T10c
            
#         svv, shh = self.Mueller2sigma(Msub) 
        svv, shh = self.Muellermod2sigma(Msub) 
        svv0, shh0 = self.sigma2sigma0([svv, shh]) 
        cpr = self.Mueller2cpr(Msub,poltype="circular")
        dlp = self.Mueller2dlp(Msub,poltype="circular")
        
        if emission == True:
            ev, eh, _, _ = self.RT0_emission("subsurface", val, thetai_rad, thetat_rad, k_a_medium, scatterer = None,  n0 = 0)
        else:
            ev = eh = 0
        
        return svv, svv0, shh, shh0, cpr, dlp, ev, eh
    
    
    
    def volumeBSC(self, val, crosspol=True, emission = True):
        self.setGeometricOptics()
        
        thetai_rad = np.deg2rad(val["thetai"])
        thetat_rad = self.TransmissionAngle(thetai_rad, val["atm_eps"], complex(val["eps1r"],val["eps1i"]))
        
        # # transmissivity into layer
        T01_coh = self.CoherentTransMatrix(val["atm_eps"], complex(val["eps1r"],val["eps1i"]), val["s1"], thetai_rad)
        T10_coh = self.CoherentTransMatrix(complex(val["eps1r"],val["eps1i"]), val["atm_eps"], val["s1"], thetat_rad)

        
        # # Extinction due to uniform background medium ulaby equation 11.47 - not necessary - taken into account in t-matrix
        k_a_medium = 0

        tm = None
        n0 = 0
        # # Create a T-matrix instance for volume scattering calculations (phase matrix and extinction matrix)
        # # Particle size distribution paramaters   
        vf = val["volfrac"]
        # # parameters for particle size distribution
        tm, n0 = self.Tmatrixcalc(self.wavelength/val["eps1r"]**0.5, val["a"],  
                                      (complex(val["eps1r"], val["eps1i"])**.5, complex(val["epsincr"], val["epsinci"])**.5), 
                                      (1-vf, vf), val["abyc"], val["alpha"], val["beta"], 
                                      psdfunc = val["psdfunc"],
                                      D_med=val["Dmax"] / 2, D_max=val["Dmax"], Lambda=val["Lambda"], mu=val["mu"], N=val["n0"]  )

        Mvol = self.Mueller_vol(val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, tm, n0)
        svv, shh = self.Muellermod2sigma(Mvol)

        svv0, shh0 = self.sigma2sigma0([svv, shh])
        cpr = self.Mueller2cpr(Mvol,poltype="circular")
        dlp = self.Mueller2dlp(Mvol,poltype="circular")
        
        if emission == True:
            ev, eh, ssa_h, ssa_v = self.RT0_emission("volume", val, thetai_rad, thetat_rad, k_a_medium, scatterer = tm,  n0 = n0)
        else:
            ev = eh = ssa_h = ssa_v = 0
        
        return svv, svv0, shh, shh0, cpr, dlp, ev, eh, ssa_h, ssa_v
    
    def volumesubBSC(self, val, crosspol=False, emission = False):
        
        self.setGeometricOptics()
        
        thetai_rad = np.deg2rad(val["thetai"])
        thetat_rad = self.TransmissionAngle(thetai_rad, val["atm_eps"], complex(val["eps1r"],val["eps1i"]))
        
        # # transmissivity into layer
        T01_coh = self.CoherentTransMatrix(val["atm_eps"], complex(val["eps1r"],val["eps1i"]), val["s1"], thetai_rad)
        T10_coh = self.CoherentTransMatrix(complex(val["eps1r"],val["eps1i"]), val["atm_eps"], val["s1"], thetat_rad)
        
        # # Extinction due to uniform background medium ulaby equation 11.47 - not necessary - taken into account in t-matrix
        k_a_medium = 0
        
        # # Reflection from subsurface interface
        R12_coh = self.CoherentRefMatrix(complex(val["eps1r"],val["eps1i"]), complex(val["eps2r"],val["eps2i"]), val["s2"], thetat_rad)

        # # Create a T-matrix instance for volume scattering calculations (phase matrix and extinction matrix)
                
        # # Particle size distribution paramaters   
        vf = val["volfrac"]
        # # parameters for particle size distribution
        tm, n0 = self.Tmatrixcalc(self.wavelength/val["eps1r"]**0.5, val["a"],  
                                      (complex(val["eps1r"], val["eps1i"])**.5, complex(val["epsincr"], val["epsinci"])**.5), 
                                      (1-vf, vf), val["abyc"], val["alpha"], val["beta"], 
                                      psdfunc = val["psdfunc"],
                                      D_med=val["Dmax"] / 2, D_max=val["Dmax"], Lambda=val["Lambda"], mu=val["mu"], N=val["n0"]  )

        Mvolsub = self.Mueller_volbed(val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12_coh, tm, n0) \
                + self.Mueller_bedvol(val, thetai_rad, thetat_rad, k_a_medium, T01_coh, T10_coh, R12_coh, tm, n0)
        svv, shh = self.Muellermod2sigma(Mvolsub)
        svv0, shh0 = self.sigma2sigma0([svv, shh])
        cpr = self.Mueller2cpr(Mvolsub,poltype="circular")
        dlp = self.Mueller2dlp(Mvolsub,poltype="circular")
        
        if emission == True:
            ev, eh, ssa_h, ssa_v = self.RT0_emission("volume", val, thetai_rad, thetat_rad, k_a_medium, scatterer = tm,  n0 = n0)
        else:
            ev = eh = ssa_h = ssa_v = 0
        
        return svv, svv0, shh, shh0, cpr, dlp, ev, eh

    
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
        ks1 = self.k * val_dict["s1"]
        ks2 = self.k * val_dict["s2"]
        shh_sur = svv_sur = cpr_sur = dlp_sur = shh_sub = svv_sub = cpr_sub = dlp_sub = shh_vol = svv_vol = cpr_vol = dlp_vol = shh_volsub = svv_volsub = cpr_volsub = dlp_volsub = shh_total = svv_total = cpr_total = dlp_total = 0
        shh0_sur = svv0_sur = shh0_sub = svv0_sub = shh0_vol = svv0_vol = shh0_volsub = svv0_volsub = shh0_total = svv0_total = np.nan
        eh_sur = ev_sur = eh_sub = ev_sub = eh_vol = ev_vol = eh_volsub = ev_volsub = eh_total = ev_total = ssa_h = ssa_v = 0

        if "surface" in scattertype:
            svv_sur, svv0_sur, shh_sur, shh0_sur, cpr_sur, dlp_sur, ev_sur, eh_sur = self.surfaceBSC(val_dict, crosspol = True, emission = True)
            
            try:
                svv_sur = svv_sur.real
                shh_sur = shh_sur.real
                svv0_sur = svv0_sur.real
                shh0_sur = shh0_sur.real
                ev_sur = ev_sur.real
                eh_sur = eh_sur.real
                cpr_sur = cpr_sur.real
                dlp_sur = dlp_sur.real
            except:
                pass
               
        if "subsurface" in scattertype:
            svv_sub, svv0_sub, shh_sub, shh0_sub, cpr_sub, dlp_sub, ev_sub, eh_sub = self.subsurfaceBSC(val_dict, crosspol = True, emission = True)
            try:
                svv_sub = svv_sub.real
                shh_sub = shh_sub.real
                svv0_sub = svv0_sub.real
                shh0_sub = shh0_sub.real
                ev_sub = ev_sub.real
                eh_sub = eh_sub.real
                cpr_sub = cpr_sub.real
                dlp_sub = dlp_sub.real
            except:
                pass
               
        if "volume" in scattertype:
            # # single scatter albedo is also returned (2 extra arguments at the end)
            svv_vol, svv0_vol, shh_vol, shh0_vol, cpr_vol, dlp_vol, ev_vol, eh_vol, ssa_h, ssa_v = self.volumeBSC(val_dict, emission = True)
            print("volume cpr = ", cpr_vol)
            try:
                svv_vol = svv_vol.real
                shh_vol = shh_vol.real
                svv0_vol = svv0_vol.real
                shh0_vol = shh0_vol.real
                ev_vol = ev_vol.real
                eh_vol = eh_vol.real
                cpr_vol = cpr_vol.real
                dlp_vol = dlp_vol.real
            except:
                pass
            
        if "volume-subsurface" in scattertype:
            svv_volsub, svv0_volsub, shh_volsub, shh0_volsub, cpr_volsub, dlp_volsub, ev_volsub, eh_volsub = self.volumesubBSC(val_dict, crosspol = True, emission = True)
            print("volsub cpr = ", cpr_volsub)
            try:
                svv_volsub = svv_volsub.real
                shh_volsub = shh_volsub.real
                svv0_volsub = svv0_volsub.real
                shh0_volsub = shh0_volsub.real
                ev_volsub = ev_volsub.real
                eh_volsub = eh_volsub.real
                cpr_volsub = cpr_volsub.real
                dlp_volsub = dlp_volsub.real
            except:
                pass

         # # total backscatter and emissivity
        shh_total = np.nansum(np.array([shh_sur , shh_sub , shh_vol , shh_volsub]))
        svv_total = np.nansum(np.array([svv_sur , svv_sub , svv_vol , svv_volsub]))
        svv0_total = self.sigma2sigma0(svv_total)
        shh0_total = self.sigma2sigma0(shh_total)
        cpr_total = np.nansum(np.array([cpr_sur, cpr_sub, cpr_vol, cpr_volsub]))
        dlp_total = np.nansum(np.array([dlp_sur, dlp_sub, dlp_vol, dlp_volsub]))
        
        
        eh_total = eh_sur + eh_sub + eh_vol + eh_volsub
        ev_total = ev_sur + ev_sub + ev_vol + ev_volsub
               
        # # make a pandas series
        cols = list(val_dict.keys())
        output_cols = ["ks1", "ks2", "shh_sur", "svv_sur", "cpr_sur", "dlp_sur", "shh_sub", "svv_sub", "cpr_sub", "dlp_sub", "shh_vol", "svv_vol", "cpr_vol", "dlp_vol", "shh_volsub", "svv_volsub", "cpr_volsub", "dlp_volsub", "shh_total", "svv_total", "cpr_total", "dlp_total", "ev_sur", "eh_sur", "ev_sub", "eh_sub", "ev_vol", "eh_vol", "ev_total", "eh_total", "ssa_v", "ssa_h"]
        cols.extend(output_cols) 
               
        values = list(val_dict.values())
        output_values = [ks1, ks2, shh0_sur, svv0_sur, cpr_sur, dlp_sur, shh0_sub, svv0_sub, cpr_sub, dlp_sub, shh0_vol, svv0_vol, cpr_vol, cpr_vol, shh0_volsub, svv0_volsub, cpr_volsub, dlp_volsub, shh0_total, svv0_total, cpr_total, dlp_total, \
                         ev_sur, eh_sur, ev_sub, eh_sub, ev_vol, eh_vol, ev_total, eh_total, ssa_v, ssa_h]
        values.extend(output_values)
        return values

    def RT0_emission(self, scattertype, val, thetai_rad, thetat_rad = 0, k_a_medium = 0, scatterer = None, n0 = 0):
        """
        Primary function that executes the oth order VRT emission model
        
        Args:
        scat: instance of class pytmatrix.Scatterer 
        
        Returns:
        e_v: Emissivity in V polarization
        e_h: Emissivity in H polarization
        """        

        e01_v, e01_h = self.I2EM_emissivity(val["thetai"], val["atm_eps"], complex(val["eps1r"],val["eps1i"]), val["s1"], val["cl1"])
        
        # # SSA = 0
        ssa_h = 0.
        ssa_v = 0.
        
        if scattertype == "surface":
            ev = e01_v
            eh = e01_h
            
        else:
            
            e12_v, e12_h = self.I2EM_emissivity(np.rad2deg(thetat_rad), complex(val["eps1r"],val["eps1i"]), complex(val["eps2r"],val["eps2i"]), val["s2"], val["cl2"])

            gamma01_v = 1 - e01_v
            gamma01_h = 1 - e01_h
            gamma12_v = 1 - e12_v
            gamma12_h = 1 - e12_h
            
            
            if scattertype == "subsurface":
#                 ev_sub = ((1 - gamma01_v) / (1 - gamma01_v * gamma12_v * trans_v**2)) * trans_v * e12_v
#                 eh_sub = ((1 - gamma01_h) / (1 - gamma01_h * gamma12_h * trans_h**2)) * trans_h * e12_h
#                 return ev_sub, eh_sub
                # # transmissivity
                trans_v = trans_h = np.exp(-k_a_medium * val["d"] / np.cos(thetat_rad))
                
           
            if scattertype == "volume":
                
                # # transmissivity 
                
#                 k_a_medium = 2 * self.k * np.sqrt(val["eps1i"])
                beta, E, Einv = self.ExtinctionMatrix(scatterer, val, n0, thetat_rad, self.phi_s)
                trans = np.linalg.multi_dot([E, self.D(beta, thetat_rad, val["d"], kab = 0), Einv])
#                 trans_v = 0.25 * (trans[0,0] + 2*trans[0,1] + trans[1,1])
#                 trans_h = 0.25 * (trans[0,0] - 2*trans[0,1] + trans[1,1])
                trans_v = trans[0,0]
                trans_h = trans[1,1]

                # # SSA (nonzero)
            
                # # resetting scatterer geometry after extinction matrix calculation
                geom = (np.rad2deg(thetat_rad), np.rad2deg(thetat_rad), np.rad2deg(self.phi_s), np.rad2deg(self.phi_s),
                        val["alpha"], val["beta"])
                scatterer.set_geometry(geom,)
                
                if scatterer.psd_integrator != None:
                    scatterer.psd_integrator.geometries = (geom,)
                    scatterer.psd_integrator.init_scatter_table(scatterer, angular_integration=True, verbose=False)
                
#                 KEH, _ = self.ExtinctionCS(scatterer, val, n0, thetat_rad, self.phi_s, pol=None)
                       
                ks_v = scatter.sca_xsect(scatterer, h_pol=False)
                ks_h = scatter.sca_xsect(scatterer, h_pol=True)          
                  
                
                ke_v = scatter.ext_xsect(scatterer, h_pol=False)
                ke_h = scatter.ext_xsect(scatterer, h_pol=True)
                
                ssa_v = scatter.ssa(scatterer, h_pol=False)
                ssa_h = scatter.ssa(scatterer, h_pol=True)  

#                 # # scattering and extinction cross sections for a sphere from Tsang 1985 and Tsang 2000
#                 k1 = self.k * val["eps1r"] ** 0.5
#                 e = complex(val["epsincr"], val["epsinci"]) / complex(val["eps1r"], val["eps1i"])
#                 y = (e-1) / (e+2)
#                 scat_cs = (8 * np.pi / 3) * k1**4 * val["a"]**6 * np.abs(y)**2
# #                 ext_cs = (4 * np.pi / k1**2) * (k1*val["a"])**3 * (y.imag + (2/3)*(k1*val["a"])**3*np.abs(y)**2)
# #                 ext_cs = scat_cs + 4 *np.pi * val["a"] ** 3 * k1*np.abs(y)**2
    
#                 # # scattering and extinction cross sections for a sphere from Ulaby
#                 ext_cs = scat_cs +  4 *np.pi * val["a"] ** 3 * k1 * e.imag / np.square(np.abs(e+2))
                 

                
#             ev_vol = ((1 - gamma01_v) / (1 - gamma01_v * gamma12_v * trans_v**2)) * (1 + gamma12_v*trans_v) * (1-a_v) * (1 - trans_v)
#             eh_vol = ((1 - gamma01_h) / (1 - gamma01_h * gamma12_h * trans_h**2)) * (1 + gamma12_h*trans_h) * (1-a_h) * (1 - trans_h)

       
            ev = ((1 - gamma01_v) / (1 - gamma01_v * gamma12_v * trans_v**2)) * \
                ((1 + gamma12_v*trans_v) * (1-ssa_v) * (1 - trans_v) + (1 - gamma12_v) * trans_v)

            eh = ((1 - gamma01_h) / (1 - gamma01_h * gamma12_h * trans_h**2)) * \
                ((1 + gamma12_h*trans_h) * (1-ssa_h) * (1 - trans_h) + (1 - gamma12_h) * trans_h)

        
        return ev, eh, ssa_h, ssa_v

       
