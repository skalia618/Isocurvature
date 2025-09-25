import numpy as np
from scipy import integrate

HI = 1e12 # in GeV
Ni = -80.
Nf = 0.
num_N = 10000
num_tau = 1000000
tauf = 3e-8 # in GeV^-1
alpha_i = 8

m = 1e8 # in GeV
lmbda = 1e-9
V = lambda phi: m ** 2 * phi ** 2 / 2 + lmbda * phi ** 4 / 24
dV = lambda phi: m ** 2 * phi + lmbda * phi ** 3 / 6
ddV = lambda phi: m ** 2 + lmbda * phi ** 2 / 2
phi_i = np.sqrt(6 * HI ** 2 * alpha_i / lmbda)
kappa = lambda phi: dddV(phi) * dV(phi) / ddV(phi) ** 2

a_rad = lambda tau: 2 + HI * tau
H_rad = lambda tau: HI / a_rad(tau) ** 2

def zero_mode_inf(y, N):
    phi, dphi = y
    return [dphi, -3 * dphi - dV(phi) / HI ** 2]

def zero_mode_rad(y, tau):
    phi, dphi = y
    return [dphi, -2 * a_rad(tau) * H_rad(tau) * dphi - a_rad(tau) ** 2 * dV(phi)]

Ns = np.linspace(Ni, Nf, num_N)
sol_inf = integrate.odeint(zero_mode_inf, [phi_i, 0], Ns)
alpha = ddV(sol_inf[:,0]) / 3 / HI ** 2

taus = np.linspace(-1 / HI, tauf, num_tau)
sol_rad = integrate.odeint(zero_mode_rad, [sol_inf[-1,0], sol_inf[-1,1] * HI], taus)

rho = lambda phi, dphi, tau: dphi ** 2 / 2 / a_rad(tau) ** 2 + V(phi)
rhos = rho(sol_rad[:,0], sol_rad[:,1], taus)

invMpc_to_GeV = 6.3949e-39
km_per_sec = 3.3356e-6
GN = 6.7088e-39 # in GeV^-2
H0 = 70 * km_per_sec * invMpc_to_GeV # in GeV
omegam = 0.32
omegac = 0.26
aeq = 1 / 3400
gstar_s = 3.9
gstar_rho = 3.4
gstar_rh = 106.75
a_today = np.sqrt(HI / H0) / (omegam * aeq) ** (1 / 4) * gstar_rh ** (1 / 12) * gstar_rho ** (1 / 4) / gstar_s ** (1 / 3)

rho_DM = 3 * omegac * H0 ** 2 / 8 / np.pi / GN
mDM = m * rho_DM / rhos[-1] * (a_today / a_rad(taus[-1])) ** 3
print('HI: {:.2e} GeV'.format(HI))
print('lambda: {:.2e}'.format(lmbda))
print('mass: {:.2e} eV'.format(mDM))

rhom = 2.5 * V(np.sqrt(12 * m ** 2 / lmbda))
am = a_rad(taus[np.argmin(np.abs(rhos - rhom))])
m_ref = 1e-8 # in GeV
HI_ref = 1e12 # in GeV
am_ref = am * HI_ref / m_ref * m / HI
kmax = np.sqrt(3) * HI_ref / am_ref / a_today / invMpc_to_GeV
print('')
print('Separate universes holds below {:.2e} Mpc^-1'.format(kmax))
print('for HI={:.2e} GeV and m={:.2e} eV'.format(HI_ref, m_ref * 1e9))

Nmin = 6
min_ind = np.argmin(np.abs(np.log(a_rad(taus)) - Nmin))
np.savez('relic', a = a_rad(taus[min_ind:]), rho = rhos[min_ind:])
