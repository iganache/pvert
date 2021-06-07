#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:39:40 2021

s = sin_theta
ss = sin_thetas
cs = cos_theta
css = cos_thetas
sf = sin(phi_s)
csf = cos(phi_s)

Kirchhoff fields are perfect.Complementary fields are wrong.

To add:
1) Multiple scattering term for backscatter (Fung, pg 250) (also in Wu and Chen)
2) Single and multiple scattering term for transmission (Fung, pg 254-255)
3) Stokes matrix changes
4) when to use eps_2 ves eps_2/eps_1
5) See if everything makes sense

@author: indujaa
"""

import numpy as np
import math
from scipy import integrate
from scipy.special import factorial, erf, erfc
from itertools import product

import FresnelCoefficients as Fresnel

debug=False

class RoughSurface:
    
    def __init__(self, wavelength, eps1 = 1, eps2 = 3, rms_ht = 0.001, corr_len = 12.6e-2, autocorr_fn = "exponential", theta_i = np.deg2rad(0.), theta_s = np.deg2rad(0)):
        
        self.sigma = rms_ht * 100        # # in centimeters
        self.corrlen = corr_len * 100    # # in centimeters
        self.autocorrelation_function = autocorr_fn
        
        self.theta_i = theta_i + 0.01
        self.theta_s = theta_i             # # use theta_i to match with Ulaby code
        
        self.phi_i = 0.
        self.phi_s = np.pi - self.phi_i
        self.phi_t = self.phi_i
        
        self.eps2 = eps2 
        self.eps1 = eps1
        self.wavelength = wavelength * 100     # # in centimeters

        self.setVariables()
        
        
    def setVariables(self):
        
        self.nu = 30 / self.wavelength        # # frequency in GHz
        self.k = 2 * np.pi / self.wavelength  # # wavenumber in 1/cm
        
        self.theta_t = Fresnel.TransmissionAngle(self.eps1, self.eps2, self.theta_i)
        
        self.k1 = self.k * np.sqrt(self.eps1)
        self.k2 = self.k * np.sqrt(self.eps2)
        
        self.kx = self.k * np.sin(self.theta_i) * np.cos(self.phi_i)
        self.ky = self.k * np.sin(self.theta_i) * np.sin(self.phi_i)
        self.kz = self.k * np.cos(self.theta_i)
        
        self.ksx = self.k * np.sin(self.theta_s) * np.cos(self.phi_s)
        self.ksy = self.k * np.sin(self.theta_s) * np.sin(self.phi_s)
        self.ksz = self.k * np.cos(self.theta_s)
        
        self.ktx = self.k * np.sin(self.theta_t) * np.cos(self.phi_t)
        self.kty = self.k * np.sin(self.theta_t) * np.sin(self.phi_t)
        self.ktz = self.k * np.cos(self.theta_t)

        self.wvnb_scat = self.k * np.sqrt((np.sin(self.theta_s)*np.cos(self.phi_s) - np.sin(self.theta_i)*np.cos(self.phi_i))**2 + \
                                (np.sin(self.theta_s)*np.sin(self.phi_s) - np.sin(self.theta_i)*np.sin(self.phi_i))**2)
        
        self.wvnb_trans = self.k * np.sqrt((np.sin(self.theta_t)*np.cos(self.phi_t) - np.sin(self.theta_i)*np.cos(self.phi_i))**2 + \
                                (np.sin(self.theta_t)*np.sin(self.phi_t) - np.sin(self.theta_i)*np.sin(self.phi_i))**2)
        
        self.qi = self.k * np.sqrt(self.eps1 - np.sin(self.theta_i)**2)
        self.qs = self.k * np.sqrt(self.eps1 - np.sin(self.theta_s)**2)
        
        self.n = self.get_n()
        self.Rv, self.Rh, self.Rvt, self.Rht = self.ReflectionCoefficients()
        self.Rvht = (self.Rvt - self.Rht) / 2
        self.Rvh = (self.Rv - self.Rh) / 2
        
    def ReflectionCoefficients(self):
        
        # # Fresnel Coefficients at incidence angle theta_i
        Rh,_,_,_ = Fresnel.FresnelH(self.eps1, self.eps2, self.theta_i)
        Rv,_,_,_ = Fresnel.FresnelV(self.eps1, self.eps2, self.theta_i)
        
        # # for consistency with Ulaby code
        Rv = -Rv
        
        # # Fresnel Coefficients at nadir incidence
        Rh0,_,_,_ = Fresnel.FresnelH(self.eps1, self.eps2, 0)
#         Rv0,_,_,_ = Fresnel.FresnelV(self.eps1, self.eps2, 0)
        Rv0 = - Rh0
        
        # # Fung has an extra step for bistatic coefficients to account for slope effects near Brewster angle. can be ignored if slope is small
        Rp_h = Rh + self.gamma_factor(Rv0) * (Rh0 - Rh)
        Rp_v = Rv + self.gamma_factor(Rv0) * (Rv0 - Rv)

        return Rv, Rh, Rp_v, Rp_h
  
        
    def gamma_factor(self, R0):

        # # check if using n is okay

        Fp = 8 * R0**2 * np.sin(self.theta_s) * \
            (np.cos(self.theta_i) + np.sqrt(self.eps2 / self.eps1 - np.sin(self.theta_i)**2)) / \
            (np.cos(self.theta_i) * np.sqrt(self.eps2 / self.eps1 - np.sin(self.theta_i)**2))


        S0 = 1 / np.abs(1 + 8*R0 / (Fp * np.cos(self.theta_i))) **2
         
        n = np.arange(1,self.n+1)
        
        Wn, rss = self.Wn(n, self.wvnb_scat)
        
        num = np.power(self.k * self.sigma * np.cos(self.theta_i), 2*n) * Wn / factorial(n) 
        den = np.power(self.k * self.sigma * np.cos(self.theta_i), 2*n) * Wn / factorial(n) * \
                np.abs(Fp + np.power(2, n+2) * R0 / np.exp((self.k*self.sigma*np.cos(self.theta_i))**2) / np.cos(self.theta_i))**2 

        Sp = np.abs(Fp) ** 2 * (np.sum(num)/np.sum(den))

        gamma_factor = 1 - (Sp/S0)
        
        return gamma_factor
        
    def Kirchhoff_Fields(self):
        " Calculates f_pq "
    
        # # Equations (22) and (23) in Wu and Chen, 2004
#         fvv = 2 * self.Rv / np.cos(self.theta_i)
#         fhh = - 2 * self.Rh / np.cos(self.theta_i)
    
        # # Equarions 2A.4 to 2A.7 from Appendix 2A in Fung (1992), page 113
        fvv = 2 * self.Rvt / (np.cos(self.theta_i)+np.cos(self.theta_s)) * \
            (np.sin(self.theta_i) * np.sin(self.theta_s) - (1 + np.cos(self.theta_i) * np.cos(self.theta_s)) * np.cos(self.phi_s - self.phi_i))
        
        fhh = -2 * self.Rht * (np.sin(self.theta_i) * np.sin(self.theta_s) - (1 + np.cos(self.theta_i) * np.cos(self.theta_s)) * \
                              np.cos(self.phi_s - self.phi_i)) / (np.cos(self.theta_i)+np.cos(self.theta_s))
        
        fhv = (self.Rvt - self.Rht) * np.sin(self.phi_s - self.phi_i)
        
        fvh = (self.Rvt - self.Rht) * np.sin(self.phi_i - self.phi_s)
             
        return fvv, fhh, fvh, fhv
    
    def Kirchhoff_Fields_trans(self):
        " Calculates f_pq "
    
        thi = self.theta_i 
        tht = self.theta_t
        phi = self.phi_i
        pht = self.phi_t

        
        mRv = 1-self.Rv
        pRv = 1+self.Rv
        mRh = 1-self.Rh
        pRh = 1+self.Rh
        
        R = self.Rvh
    
        mR = 1-R
        pR = 1+R
        
        # # Fung 1994 - Appendix 4D - equations 4D.1 - 4D.3
        
        eps_r = self.eps2 / self.eps1
        eta_r = 1 / np.sqrt(eps_r)
        
        Zx = (np.sqrt(eps_r) * np.sin(tht) * np.cos(pht-phi) - np.sin(thi)) / (np.sqrt(eps_r)*np.cos(tht) - np.cos(thi))
        Zy = (np.sqrt(eps_r) * np.sin(tht) * np.sin(pht-phi)) / (np.sqrt(eps_r)*np.cos(tht) - np.cos(thi))
        
        ftvv = mRv*((np.cos(thi) + Zx*np.sin(thi)) * np.cos(pht-phi) + Zy*np.sin(thi)*np.sin(pht-phi)) \
            + pRv * (np.cos(tht) * np.cos(pht-phi) + Zx*np.sin(tht)) * eta_r
        
        fthh = - pRh * (np.cos(tht) * np.cos(pht-phi) + Zx*np.sin(tht)) \
            - mRh * ((np.cos(thi) + Zx*np.sin(thi)) * np.cos(pht-phi) + Zy*np.sin(thi)*np.sin(pht-phi)) * eta_r
        
        fthv = mR * ((np.cos(thi) + Zx*np.sin(thi)) * np.cos(tht)*np.sin(pht-phi) \
                     + Zy*(np.sin(tht) * np.cos(thi) - np.sin(thi)*np.cos(tht)*np.cos(pht-phi))) + pR * eta_r * np.sin(pht-phi)
        
        ftvh = mR * np.sin(pht-phi) + pR*eta_r* ((np.cos(thi) + Zx*np.sin(thi))* np.cos(tht) * np.sin(pht-phi) \
                                                + Zy* (np.sin(tht)*np.cos(thi) - np.sin(thi) * np.cos(tht) * np.cos(pht-phi)))
        
        return ftvv, fthh, ftvh, fthv

    def Copol_BSC(self):
        """
        Compute the rough surface backscatter coefficient using Fung's I2EM formula
        
        Returns:
        sigmavv: Backscatter coefficient for VV polarization 
        sigmahh: Backscatter coefficient for HH polarization 
        """
        
        n = np.arange(1,self.n+1)
        Wn, rss = self.Wn(n, self.wvnb_scat)

        
        shdw = self.Shadow_fn(mode="single")
        loss_factor = 0.5 * self.k**2 * np.exp(-self.sigma**2 *(self.kz**2 + self.ksz**2))
        
        sigmavv = loss_factor * shdw * np.sum(np.power(self.sigma, 2*n) * np.abs(self.I_qp_copol(n)[0])**2 * Wn / factorial(n) )
        sigmahh = loss_factor * shdw * np.sum(np.power(self.sigma, 2*n) * np.abs(self.I_qp_copol(n)[1])**2 * Wn / factorial(n) )
        
        return sigmavv, sigmahh
    
    def Copol_BSC_trans(self):
        """
        Compute the rough surface transmitted scattering coefficient using Fung's I2EM formula
        
        Returns:
        sigmavv_trans: Transmitted scattering coefficient for VV polarization 
        sigmahh_trans: Transmitted scattering coefficient for HH polarization 
        """
        
        n = np.arange(1,self.n+1)
        Wn, rss = self.Wn(n, self.wvnb_trans)

        
        loss_factor = 0.5 * np.abs(self.k2)**2 * np.exp(-self.sigma**2 *(self.kz**2 + self.ktz.real**2)) 
        
        sigmavv_trans = loss_factor * np.sum(np.power(self.sigma, 2*n) * np.abs(self.I_qp_copol_trans(n)[0])**2 * Wn / factorial(n) )
        sigmahh_trans = loss_factor * np.sum(np.power(self.sigma, 2*n) * np.abs(self.I_qp_copol_trans(n)[1])**2 * Wn / factorial(n) )
        
        return sigmavv_trans, sigmahh_trans
    
    def Crosspol_BSC(self):
        """
        uses double quad to integrate over r and phi in the place of u and v
        Figure out including HV and VH
        """ 

        sigmavh, err = integrate.dblquad(self.crosspol_integ, 0, np.pi ,lambda r: 0.1, lambda r:1)
        sigmavh *= sigmavh * 1e-5 * self.Shadow_fn("multi")                     # # re-rescale after dblquad
        
#         sigmavh = integrate.dblquad(self.crosspol_integ, 0, np.pi ,lambda r: 0.1, lambda r:1)
        
        return sigmavh
        
    def crosspol_integ(self, r, phi, pol="HV"):
        """
        Possibly normalized to k
        """
        
        n = np.arange(1,self.n+1)
        m = np.arange(1,self.n+1)
        idx = np.array(list(product(m,n)))
        
        Wn, rss_n = self.Wn_multi_scat(n, r, phi)
        Wm, rss_m = self.Wn_multi_scat(m, r, phi)
        
#         shdw = self.Shadow_fn_crosspol()
        
#         loss_factor = self.k**2 * np.exp(-self.sigma**2 *(self.kz**2 + self.ksz**2)) / 16 / np.pi                              # # not normalized to k
        loss_factor = np.exp(-(self.k*self.sigma)**2 *(np.cos(self.theta_i)**2 + np.cos(self.theta_s)**2)) / 16 / np.pi                   # # normalized to k
    
        shdw = self.Shadow_fn_multi(r, phi)
        Fhv, Fvh = self.crosspolComplementaryFields(r, phi)
        
        
        comp = 0
        
        for i in n:
            for j in m:
#                 comp = comp + np.power(self.kz**2 * self.sigma**2, i+j) * Wn(i) * Wm(j) / factorial(i) / factorial(j)                        # # not normalized to k
                comp = comp + np.power(np.cos(self.theta_i)**2 * (self.k * self.sigma)**2, i+j) * Wn[i-1] * Wm[j-1] / factorial(i) / factorial(j)             # # not normalized to k
    
#         sigma_unint = loss_factor * (np.abs(Fvh) ** 2 + Fvh*Fvh.conjugate()) * comp * shdw
        sigma_unint = 4 * loss_factor * Fvh * comp * r * shdw
         
#         print(idx[:,1], idx[:,1][0], idx[:,1]-1)
#         comp = np.sum(np.power(np.cos(self.theta_i)**2 * self.sigma**2, idx[:,0]+idx[:,1]) * Wn(idx[:,0]) * Wm(idx[:,1]) / factorial(idx[:,0]) / factorial(idx[:,1]))
        
#         if pol == "HV":
#             sigma_unint = loss_factor * (np.abs(Fhv) ** 2 + Fhv*Fhv.conjugate()) * comp
#         elif pol == "VH":
#             sigma_unint = loss_factor * (np.abs(Fvh) ** 2 + Fvh*Fvh.conjugate()) * comp
            
        sigma_unint *= 1e5                                  # # rescaling to make dblquad owrk better
        return sigma_unint

    
    def I_qp_copol(self, n):
        
        fvv, fhh, fvh, fhv = self.Kirchhoff_Fields()      

        Fvv_up_inc, Fhh_up_inc = self.ComplementaryFields(1, 1, -self.kx, -self.ky, self.theta_i)
        Fvv_down_inc, Fhh_down_inc = self.ComplementaryFields(-1, 1, -self.kx, -self.ky, self.theta_i)
        Fvv_up_scat, Fhh_up_scat = self.ComplementaryFields(1, 2, -self.ksx, -self.ksy, self.theta_s)
        Fvv_down_scat, Fhh_down_scat = self.ComplementaryFields(-1, 2, -self.ksx, -self.ksy, self.theta_s)

        
        Ivv = np.power(self.kz+self.ksz, n) * fvv * np.exp(-self.sigma**2 * self.kz * self.ksz)\
            + 0.25 * (np.power(self.ksz-self.qi, n-1) * Fvv_up_inc * np.exp(-self.sigma**2 * (self.qi**2 - self.qi*self.ksz + self.qi*self.kz)) + \
                      np.power(self.ksz+self.qi, n-1) * Fvv_down_inc * np.exp(-self.sigma**2 * (self.qi**2 + self.qi*self.ksz - self.qi*self.kz)) + \
                      np.power(self.kz+self.qs, n-1) * Fvv_up_scat * np.exp(-self.sigma**2 * (self.qs**2 - self.qs*self.ksz + self.qs*self.kz)) +\
                      np.power(self.ksz-self.qs, n-1) * Fvv_down_scat * np.exp(-self.sigma**2 * (self.qs**2 + self.qs*self.ksz - self.qs*self.kz)))
        
        Ihh =  np.power(self.kz+self.ksz, n) * fhh * np.exp(-self.sigma**2 * self.kz * self.ksz)\
            + 0.25 * (np.power(self.ksz-self.qi, n-1) * Fhh_up_inc * np.exp(-self.sigma**2 * (self.qi**2 - self.qi*self.ksz + self.qi*self.kz)) + \
                      np.power(self.ksz+self.qi, n-1) * Fhh_down_inc * np.exp(-self.sigma**2 * (self.qi**2 + self.qi*self.ksz - self.qi*self.kz)) + \
                      np.power(self.kz+self.qs, n-1) * Fhh_up_scat * np.exp(-self.sigma**2 * (self.qs**2 - self.qs*self.ksz + self.qs*self.kz)) +\
                      np.power(self.ksz-self.qs, n-1) * Fhh_down_scat * np.exp(-self.sigma**2 * (self.qs**2 + self.qs*self.ksz - self.qs*self.kz)))

        
        return Ivv, Ihh
    
    
    def I_qp_copol_trans(self, n):
        
        ftvv, fthh, ftvh, fthv = self.Kirchhoff_Fields_trans()      

        Ftvv_inc, Fthh_inc = self.ComplementaryFields_trans(1, -self.kx, -self.ky, self.theta_i)
        Ftvv_trans, Fthh_trans = self.ComplementaryFields_trans(2, -self.ktx, -self.kty, self.theta_t)
        
        Ivv = np.power(self.kz-self.ktz.real, n) * ftvv * np.exp(-self.sigma**2 * self.kz * self.ktz.real) \
            + 0.5 * ( np.power(self.ktz.real, n-1) * Ftvv_inc +  np.power(self.kz, n-1) * Ftvv_trans)
        
        Ihh = np.power(self.kz-self.ktz.real, n) * fthh * np.exp(-self.sigma**2 * self.kz * self.ktz.real) \
            + 0.5 * ( np.power(self.ktz.real, n-1) * Fthh_inc +  np.power(self.kz, n-1) * Fthh_trans)
        
        return Ivv, Ihh

        
    def ComplementaryFields(self, updown, method, u, v, theta):
        """
        Calculates Fvv and Fhh 
        """ 
        
        # # Equations (24) and (25) in Wu and Chen, 2004
#         Fvv = 2 * np.sin(theta_i)**2 / np.cos(theta_i) * (1 + Rv**2) * ((self.eps2 - 1) * (np.sin(theta_i) / np.cos(theta_i) / self.eps2) **2  + (1 - 1/self.eps2))
#         Fhh = 2 * np.sin(theta_i)**2 / np.cos(theta_i)**3 * (self.eps2 - 1)
        
        eps_r = self.eps2 / self.eps1

        q1 = self.k * np.sqrt(self.eps1 - np.sin(theta)**2)
        q2 = self.k * np.sqrt(self.eps2 - np.sin(theta)**2)
        
        mRv = 1-self.Rvt
        pRv = 1+self.Rvt
        mRh = 1-self.Rht
        pRh = 1+self.Rht

        
        # # C functions
        
#         np.set_printoptions(formatter={'complex_kind': '{:.4f}'.format},suppress = True)
        C1, C2, C3, C4, C5 = self.C_functions(updown, method, q1)
        C1t, C2t, C3t, C4t, C5t = self.C_functions(updown, method, q2)


#         C1, C2, C3, C4, C5, C1t, C2t, C3t, C4t, C5t = self.C_functions_pyrism(updown, method)


        Fvv = -(mRv/q1*C1 - pRv/q2*C1t) * pRv + (mRv/q1*C2 - pRv/q2*C2t) * mRv + (mRv/q1*C3 - pRv/q2/eps_r*C3t) * pRv +\
                (pRv/q1*C4 - mRv*eps_r/q2*C4t) * mRv + (pRv/q1*C5 - mRv/q2*C5t) * pRv 

                                              
        Fhh = (mRh/q1*C1 - pRh*eps_r/q2*C1t) * pRh - (mRh/q1*C2 - pRh/q2*C2t) *mRh - (mRh/q1*C3 - pRh/q2*C3t) *pRh -\
                (pRh/q1*C4 - mRh/q2*C4t) * mRh - (pRh/q1*C5 - mRh/q2*C5t) * pRh 


        return Fvv, Fhh
    
    def ComplementaryFields_trans(self, method, u, v, theta):
        eps_r = self.eps2 / self.eps1

        q1 = self.k * np.sqrt(self.eps1 - np.sin(self.theta_i)**2)
        q2 = self.k * np.sqrt(self.eps2 - np.sin(self.theta_t)**2)
        
#         q1 = self.k * np.sqrt(self.eps1 - np.sin(theta)**2)
#         q2 = self.k * np.sqrt(self.eps2 - np.sin(theta)**2)
 
        eta_r = 1/np.sqrt(eps_r)
        
        mRv = 1-self.Rv
        pRv = 1+self.Rv
        mRh = 1-self.Rh
        pRh = 1+self.Rh
        
        # # C functions
        
#         np.set_printoptions(formatter={'complex_kind': '{:.4f}'.format},suppress = True)
        C1, C2, C3, C4, C5, C6 = self.C_functions_trans(method)
        
        Ftvv = -(mRv/q1 - pRv/q2) *C1 * pRv + (mRv/q1 - pRv/q2)*C2 * mRv + (mRv/q1 - pRv/eps_r/q2) *C3* pRv \
            + (pRv/q1 - mRv/q2*eps_r) *C4 * mRv*eta_r + (pRv/q1 - mRv/q2)*C5 *pRv * eta_r + (pRv/q1 - mRv/q2)*C6 *mRv*eta_r
        
        Fthh = (mRh/q1 - pRh*eps_r/q2)*C1 * pRh*eta_r - (mRh/q1 - pRh/q2)*C2 * mRh* eta_r - (mRh/q1 - pRh/q2)*C3 * pRh*eta_r \
            - (pRh/q1 - mRh/q2)*C4 * mRh - (pRh/q1 - mRh/q2)*C5 * pRh - (pRh/q1 - mRh/eps_r/q2)*C6 * mRh

        return Ftvv, Fthh
 
    
    def crosspolComplementaryFields(self, r, phi):
        
        """
        Fhv and Fvh from Fung 1994 - Page 201,202 - eq 4.B.19 and 4.B.20
        Rewriting in terms of r = k*sin(theta_i) and phi
        Is r (and everything else) normalized to k?
        """
        
        eps_r = self.eps2 / self.eps1
        
        q1 = np.sqrt(self.eps1 - r**2)
        q2 = np.sqrt(self.eps2 - r**2)

        
        R = self.Rvht
    
        mR = 1-R
        pR = 1+R
        
        # # B functions
        B1, B2, B3, B4, B5, B6 = self.B_functions(r, phi)
        
        Fhv = (mR/q1 - pR/q2) * pR*B1 - (mR/q1 - pR/q2) * mR * B2 - (mR/q1 - pR/eps_r/q2) \
            - (pR/q1 - mR*eps_r/q2) * mR*B4 + (pR/q1 - mR/q2) * mR*B4 + (pR/q1 - mR/q2) * mR*B6
        
        Fvh = (pR/q1 - mR*eps_r/q2) * mR*B1 - (pR/q1 - mR/q2) *pR*B2 - (pR/q1 - mR/q2) *mR*B3 \
            + (mR/q1 - pR/q2) * pR*B4 + (mR/q1 - pR/q2) * mR*B5 + (mR/q1 - pR/eps_r/q2) * pR*B6     
        
        q = np.sqrt(1.0001 - r**2);
        qt = np.sqrt(eps_r - r**2);

        rm = 1-R
        rp = 1+R
        a = rp /q
        b = rm /q
        c = rp /qt
        d = rm /qt

        B3 = r*np.cos(phi) * r*np.sin(phi) /np.cos(self.theta_i)
        fvh1 = (b-c)*(1- 3*R) - (b - c/eps_r) * rp; 
        fvh2 = (a-d)*(1+ 3*R) - (a - d*eps_r) * rm;
        Fvh = ( np.abs( (fvh1 + fvh2) *B3))**2;
        
        return Fhv, Fvh
        

    def Wn(self, n, wvnb):
        """
        W(n) is the the spectral power density aka Fouier transform of the nth power of the surface correlation function
        Current formulae from Fung 1992, Appendix 2B, page 117 to 119
        """
        # # currently has exponential and gaussian. expand to include 2d gaussian, 2d exponential and 1.5 power (eqs 4a-4f in Brogioni et al. 2010)
              
        lc = self.corrlen
        
        if self.autocorrelation_function == "gaussian":
            # gaussian C(r) = exp ( -(r/l)**2 )
            wn = (lc**2 / (2 * n)) * np.exp(-(wvnb * lc)**2 / (4 * n))
            rss = np.sqrt(2) * self.sigma / lc
        elif self.autocorrelation_function == "exponential":
            # exponential C(r) = exp( -r/l )
            wn = (lc / n)**2 * (1 + (wvnb * lc / n)**2)**(-1.5)
            rss = self.sigma / lc
        return wn, rss
    
    def Wn_multi_scat(self, n, r, phi):
        
        kl = self.k * self.corrlen
        
        if self.autocorrelation_function == "gaussian":
            # gaussian C(r) = exp ( -(r/l)**2 )
            wn = (kl**2 / (2 * n)) * np.exp(-kl**2 * ((r*np.cos(phi) - np.sin(self.theta_i)) ** 2 + (r*np.sin(phi)) ** 2) / (4 * n))
            rss = np.sqrt(2) * self.sigma / self.corrlen
        elif self.autocorrelation_function == "exponential":
            # exponential C(r) = exp( -r/l )
            wn = (n * kl **2) / (n ** 2 + kl**2 * ((r*np.cos(phi) - np.sin(self.theta_i)) ** 2 + (r*np.sin(phi)) ** 2))**(1.5)
            rss = self.sigma / self.corrlen
            
        return wn, rss
    
    def get_n(self):
        error = 1e8

        Ts = 1
        while error > 1e-3:
            Ts += 1
            error = ((self.k * self.sigma) ** 2 * ( np.cos(self.theta_i)+np.cos(self.theta_s) ) **2) ** Ts / factorial(Ts)

        return Ts
    
    def Shadow_fn(self, mode="multi"):
        
        # # From Fung textbook
        n = np.arange(1,self.n+1)
        Wn, rss = self.Wn(n, self.wvnb_scat)
        
        ct_i = 1 / (np.tan(self.theta_i) * np.sqrt(2) * rss)
        ct_s = 1 / (np.tan(self.theta_s) * np.sqrt(2) * rss)
        
        shdwf = 0.5 * (np.exp(-1 * ct_i**2) / np.sqrt(np.pi) / ct_i - erfc(ct_i))
        shdws = 0.5 * (np.exp(-1 * ct_s**2) / np.sqrt(np.pi) / ct_s - erfc(ct_s))
        
        if mode == "single":
            shdw = 1 / (1 + shdwf + shdws)
        elif mode == "multi":
            shdw = 1 / (1 + shdwf)
        return shdw
    
    def Shadow_fn_multi(self, r, phi):
        # # From Ulaby code
        n = np.arange(1,self.n+1)
        Wn, rss = self.Wn(n, self.wvnb_scat)
        
        q1 = np.sqrt(self.eps1 - r**2)
        au = q1 / (r*np.sqrt(2)*rss)
        fsh = (0.2821/au) *np.exp(-au**2) -0.5 *(1- erf(au))
        shdw = 1 / (1+fsh)

        return shdw
        
    
    def C_functions(self, updown, method, q):

        qi = updown * self.qi
        qs = updown * self.qs
        q = updown * q
        
        thi = self.theta_i 
        ths = self.theta_s
        phi = self.phi_i
        phs = self.phi_s 
        
        kszq = self.ksz - qi
        kzq = self.kz + qs
        
        if method == 1:
            C1 = self.k * np.cos(phs) * kszq
            C2 = np.cos(thi) * (np.cos(phs) * (self.k ** 2 * np.sin(thi) * np.cos(phi) * (np.sin(ths) * np.cos(phs) - np.sin(thi)* np.cos(phi)) + q*kszq) \
                               + self.k**2 * np.cos(phi) * np.sin(thi) *np.sin(ths) * np.sin(phs)**2)
            C3 = self.k * np.sin(thi) * (np.sin(thi) * np.cos(phi) * np.cos(phs) * kszq \
                                         - q * (np.cos(phs) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi)) + np.sin(ths) * np.sin(phs)**2))
            C4 = self.k * np.cos(thi) * (np.cos(ths) * np.cos(phs) * kszq + self.k * np.sin(ths) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi)))
            
            C5 = q * (np.cos(ths) * np.cos(phs) * -kszq - self.k * np.sin(ths) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi)))

        if method == 2:
            C1 = self.k * np.cos(phs) * kzq
            C2 = q * (np.cos(phs) * (np.cos(thi) * kzq - self.k * np.sin(thi) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi))) - \
                     self.k * np.sin(thi) * np.sin(ths) * np.sin(phs) **2)
            C3 = self.k * np.sin(ths) * (self.k * np.cos(thi) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi)) + np.sin(thi)*kzq)
            
            C4 = self.k * np.cos(ths) * (np.cos(phs) * (np.cos(ths) * kzq - self.k * np.sin(thi) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi))) -\
                                        self.k * np.sin(thi) * np.sin(ths) * np.sin(phs)**2)
            C5 = -np.cos(ths) * (self.k**2 * np.sin(ths) * (np.sin(ths) * np.cos(phs) - np.sin(thi) * np.cos(phi)) + q * np.cos(phs) * kzq)
          
        
        return C1, C2, C3, C4, C5
    
    def C_functions_trans(self, method):
        
        thi = self.theta_i 
        tht = self.theta_t
        phi = self.phi_i
        pht = self.phi_t
        
        eps_r = self.eps2/self.eps1
        
        # # Fung 1994, Appendix 4D, equation 4.B.20 - 4.B.30
        
        if method == 1:
            Ct1 = self.k * np.cos(pht - phi)
            Ct2 = self.k * np.sin(thi) * np.cos(thi) / np.cos(tht) * (np.sin(thi)*np.cos(pht-phi)/np.sqrt(eps_r) - np.sin(tht))
            Ct3 = self.k * np.sin(tht) ** 2 * np.cos(pht-phi)
            Ct4 = self.k * np.cos(thi) / np.cos(tht) * ( np.sin(thi)*np.sin(tht)/np.sqrt(eps_r) - np.cos(pht-phi) )
            Ct5 = Ct6 = 0
        
        if method == 2:
            Ct1 = self.k * np.cos(pht - phi)
            Ct3 = self.k * eps_r * np.sin(tht) ** 2 * np.cos(pht-phi)
            Ct4 = self.k * np.cos(tht) / np.cos(thi) * (np.sqrt(eps_r)*np.sin(thi) * np.sin(tht) - np.cos(pht-phi))
            Ct5 = self.k * np.sqrt(eps_r) * np.sin(tht) * np.cos(tht) / np.cos(thi) * (np.sqrt(eps_r) * np.sin(tht) * np.cos(pht-phi) - np.sin(thi))
            Ct2 = Ct6 = 0
            
        return Ct1, Ct2, Ct3, Ct4, Ct5, Ct6
            

    
    def B_functions(self, r, phi):
        
        """
        computes the co-polarization scattering functions needed for determining the complementary fields
        Uses equations 4.B.21 to 4.B.32 in Fung 1994 book
        Also equations A.29 to A.34 in Fung 1994 paper
        rewrite in terms of r=k*sin(theta_i) and phi
        Looks like everything gets normalized to k
        """
        
        denom = self.k * np.cos(self.theta_i)
        denom = denom/self.k
        
        u = r * np.cos(phi)
        v = r * np.sin(phi)

        B3 = (u*v) / denom
        B1 = B4 = B6 = - B3
        B2 = - 2 * B3
        B5 = 2 * B3
        
        return B1, B2, B3, B4, B5, B6
    
    def B_functions_trans(self):
        pass
                
             