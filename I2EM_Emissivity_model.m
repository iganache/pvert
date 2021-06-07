%% Source: Microwave Radar and Radiometric Remote Sensing, http://mrs.eecs.umich.edu
%% These MATLAB-based computer codes are made available to the remote
%% sensing community with no restrictions. Users may download them and
%% use them as they see fit. The codes are intended as educational tools
%% with limited ranges of applicability, so no guarantees are attached to
%% any of the codes. 
%%
%Code 12.2: I2EM Rough Surface Emissivity

%Description: Code computes the emissivity of a homogeneous medium with
%rough surface characterized by either the exponential correlation function
%or the Gaussian correlation function. 

%Input variables:
    %er: Dielectric constant of the surface
    %theta_d: incidence angle (degrees)
    %sig: rms height (m)
    %L: correlation length (m)
    %fr: frequency in GHz
    %sp: 1- exponential and 2- gaussian correlation function

%Output Products:
    %e_h and e_v: h- and v-polarized emissivities
    
% Book Reference: Section 12-3.1

%Matlab code:

function [ev, eh] = I2EM_Emissivity_model(fr, sig, L, theta_d, er, sp)

%-- This code calculates the emission from rough surfaces using the I2EM
%model

sig = sig * 100; % transform to cm
L = L* 100; %transform to cm.


error = 1.0e8;

k = 2*pi *fr/30; % wavenumber in free space. Speed of light is in cm/sec
theta = theta_d .*pi/180; % transform to radian

ks = k * sig; % roughness parameter
kl = k * L;  

ks2 = ks .* ks; 
kl2 = kl.^2;

cs = cos(theta);
s = sin(theta);

s2 = s.^2;

%-- calculation of reflection coefficints
sq = sqrt(er - s2);

rv0 = (sqrt(er) - 1) ./(sqrt(er) + 1);
rh0 = -rv0;

rv = (er *cs - sq) ./(er*cs +sq);
rh = (cs - sq)./(cs + sq);



pol = 1;
refv = dblquad(@(ths,phs)emsv_integralfunc(ths, phs, pol, sp, k, L, ks, cs,s, kl, er, sq,rv,rh), 0.0, pi/2, 0, pi);

pol = 2;
refh = dblquad(@(ths,phs)emsv_integralfunc(ths, phs, pol, sp, k, L, ks, cs,s, kl, er, sq,rv,rh), 0.0, pi/2, 0, pi);

ev = 1- refv - exp(-ks2 .* cs.*cs) .* (abs(rv)).^2;
eh = 1- refh - exp(-ks2 .* cs.*cs) .* (abs(rh)).^2;

end

function ref = emsv_integralfunc(ths, phs, pol, sp, k,L, ks, cs,s, kl, er, sq, rv,rh)

error = 1.0e3;

cs2 = cs .^2;
ks2 = ks.^2;
s2 = s .*s;
kl2 = kl.* kl;

css = cos(ths);
css2 = css .* css;

ss = sin(ths);
ss2 = ss .* ss;

sf = sin(phs);
sf2 = sf .*sf;
csf= cos(phs);

sqs = sqrt(er - ss2);
rc = (rv - rh) ./2;
tv = 1 + rv;
th = 1 + rh; 
tv2 = tv .* tv;
th2 = th .* th;

%-- calc coefficients for surface correlation spectra
wvnb = k .* sqrt(s2 - 2 .* s .*ss .* csf + ss2); 

nr = length(ths);
thsmin = min(ths); % selected as the smallest angle in ths vector
costhsmin = cos(thsmin); 

%-- calculate number of spectral components needed 
n_spec = 1; 
while error > 1.0e-3
  n_spec = n_spec + 1;
%   error = (ks2 .*(cs + css).^2 ).^n_spec ./ factorial(n_spec); 
%---- in this case we will use the smallest ths to determine the number of
%spectral components to use. It might be more than needed for other angles
%but this is fine. This option is used to simplify calculations.
  error = (ks2 .*(cs + costhsmin).^2 ).^n_spec ./ factorial(n_spec); 
end


%-- calculate expressions for the surface spectra
wn = spectrm(sp, kl2, L, wvnb, n_spec, nr);

%-- calculate fpq!

ff = 2 .*(s .* ss - (1 + cs .* css) .* csf)./(cs + css);

fvv = rv .* ff;
fhh = -rh .* ff;

fvh = -2 * rc .* sf;
fhv = 2 .* rc .* sf;

%-- calculate Fpq and Fpqs -----
cm1 = s .*(ss - s .*csf) ./(cs2 .* css);
T = (sq .*(cs + sq) + cs.*(er.*cs + sq)) ./ (er .*cs .*(cs+sq) + sq .*(er.*cs +sq));
cm2 = css .* sq ./cs ./sqs - 1;
ex = exp(-ks2 .*cs .*css);
de = 0.5 .*exp(-ks2 .*(cs2 + css2)); 

if pol == 1
  Fvv = (er - 1) .*s2 .*tv2 .*cm1 ./ er.^2;
  Fhv = (T .*s .*s -1.+ cs./css + (er .*T .*cs .*css .*(er .*T - s .*s) - ...
     sq .* sq) ./ (T .*er .*sq .* css)) .*(1 - rc .*rc) .*sf;
 
  Fvvs = -cm2 .*sq .*tv2 .*(csf - s .*ss) ./cs2 ./er - cm2 .* sqs .*tv2 .*csf ./er ...
    -(css .*sq ./cs ./sqs ./er -1) .* ss .* tv2 .* (s - ss .*csf) ./cs;
  Fhvs = -(ss.*ss ./T -1 + css ./cs + (cs .*css .*(1- ss.*ss.*T) - ...
    T .*T.*sqs.*sqs) ./(T .*sqs.*cs)).*(1-rc .*rc).*sf;

   %-- calculate the bistatic field coefficients ---

  svv = zeros(n_spec, nr); 
  for n = 1:n_spec
    Ivv = fvv .*ex .*(ks .*(cs + css)).^n + (Fvv .*(ks .*css).^n + Fvvs .*(ks.*cs).^n) ./2;
    Ihv = fhv .*ex .*(ks .*(cs + css)).^n + (Fhv .*(ks .*css).^n + Fhvs .*(ks.*cs).^n) ./2;

    wnn = wn(n,:)./factorial(n);
    vv = wnn .* (abs(Ivv)).^2;
    hv = wnn .* (abs(Ihv)).^2;
    svv(n,:) = de .*(vv + hv) .* ss ./4/pi ./cs;
  end    

ref = sum(svv,1); % adding all n terms stores in different rows
end

if pol == 2
  Fhh = -(er -1) .* th2 .* cm1;
  Fvh = (s .*s ./T -1.+ cs./css + (cs .*css .*(1- s .*s .*T) - ...
    T .*T .*sq .* sq) ./ (T .*sq .* css)) .*(1 - rc .*rc) .*sf;

  Fhhs = cm2 .* sq .*th2 .*(csf - s .*ss) ./cs2 + cm2 .* sqs .* th2 .* csf + ...
     cm2 .*ss .*th2 .*(s - ss.*csf)./cs;
  Fvhs = -(T .* ss .* ss - 1 + css./cs + (er .*T .*cs .*css .*(er.*T - ss .*ss) ...
    - sqs .* sqs) ./(T .* er .*sqs .*cs)) .*(1- rc .*rc).*sf;


  for n = 1:n_spec
   Ihh = fhh .*ex .*(ks .*(cs + css)).^n + (Fhh .*(ks .*css).^n + Fhhs .*(ks.*cs).^n) ./2;
   Ivh = fvh .*ex .*(ks .*(cs + css)).^n + (Fvh .*(ks .*css).^n + Fvhs .*(ks.*cs).^n) ./2;
  
    wnn = wn(n,:)./factorial(n);
    hh = wnn .* (abs(Ihh)).^2;
    vh = wnn .* (abs(Ivh)).^2;
    shh(n,:) = de .*(hh + vh) .* ss ./4/pi ./cs;
  end    
  
ref = sum(shh, 1); % adding all n terms stores in different rows
end

end

%------------------------------------------------------------------
%------------------------------------------------------------------
function wn = spectrm(sp, kl2, L, wvnb, np, nr)

wn = zeros(np,nr);

if sp == 1  % exponential
    for n = 1: np       
        wn(n, :) = n* kl2 ./(n.^2 + (wvnb.*L).^2).^1.5;
    end
end

if sp == 2  %  gaussian
    for n = 1: np
        wn(n,:) = 0.5 * kl2 ./n .* exp(-(wvnb .*L).^2/(4*n)) ;
    end
end

end
