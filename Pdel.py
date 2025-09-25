import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

HI = 1e12 # in GeV
Ni = -80.
Nf = 0.
num0 = 10000
num_tau = 10000
numk = 10000
alpha_i = 8

lmbda = 1e-9
V = lambda phi: lmbda * phi ** 4 / 24
dV = lambda phi: lmbda * phi ** 3 / 6
ddV = lambda phi: lmbda * phi ** 2 / 2
phi_i = np.sqrt(6 * HI ** 2 * alpha_i / lmbda)
filename = 'Pdel_p4'

##p = 2.1
##Lambda = 4.7e11 # in GeV
##V = lambda phi: Lambda ** (4 - p) * np.abs(phi) ** p
##dV = lambda phi: np.sign(phi) * p * Lambda ** (4 - p) * np.abs(phi) ** (p - 1)
##ddV = lambda phi: p * (p - 1) * Lambda ** (4 - p) * np.abs(phi) ** (p - 2)
##phi_i = (3 * HI ** 2 * alpha_i / p / (p - 1) / Lambda ** (4 - p)) ** (1 / (p - 2))
##filename = 'Pdel_p2.1'

def zero_mode(y, N):
    phi, dphi = y
    return [dphi, -3 * dphi - dV(phi) / HI ** 2]

Ns = np.linspace(Ni, Nf, num0)
sol0 = integrate.odeint(zero_mode, [phi_i, 0], Ns)
alpha = ddV(sol0[:,0]) / 3 / HI ** 2

a = lambda tau: -1 / HI / tau
N_to_tau = lambda N: -1 / HI / np.exp(N)

rho = lambda phi, dphi: HI ** 2 * dphi ** 2 / 2 + V(phi)
delta = lambda phi0, dphi0, phi, dphi: (HI ** 2 * dphi0 * dphi + dV(phi0) * phi) / rho(phi0, dphi0)

def delsqr(k):
    tau0 = -30 / k # in GeV^-1
    tauf = -0.01 / k # in GeV^-1

    taus = np.linspace(tau0, tauf, num_tau)
    Ns_k = np.linspace(np.log(a(tauf)), Nf, numk)

    def k_mode_tau(y, tau):
        phi, dphi = y
        phi0 = np.interp(tau, N_to_tau(Ns), sol0[:,0])
        return [dphi, -2 * a(tau) * HI * dphi - (k ** 2 + a(tau) ** 2 * ddV(phi0)) * phi]

    solkR_tau = integrate.odeint(k_mode_tau, [np.cos(k * tau0) / np.sqrt(2 * k) / a(tau0), -(a(tau0) * HI * np.cos(k * tau0) + k * np.sin(k * tau0)) / np.sqrt(2 * k) / a(tau0)], taus)
    solkI_tau = integrate.odeint(k_mode_tau, [-np.sin(k * tau0) / np.sqrt(2 * k) / a(tau0), (a(tau0) * HI * np.sin(k * tau0) - k * np.cos(k * tau0)) / np.sqrt(2 * k) / a(tau0)], taus)
    solk_tau = solkR_tau + 1j * solkI_tau

    def k_mode_N(y, N):
        phi, dphi = y
        phi0 = np.interp(N, Ns, sol0[:,0])
        return [dphi, -3 * dphi - (k ** 2 / np.exp(2 * N) + ddV(phi0)) / HI ** 2 * phi]

    solkR_N = integrate.odeint(k_mode_N, [solkR_tau[-1,0], solkR_tau[-1,1] / a(tauf) / HI], Ns_k)
    solkI_N = integrate.odeint(k_mode_N, [solkI_tau[-1,0], solkI_tau[-1,1] / a(tauf) / HI], Ns_k)
    solk_N = solkR_N + 1j * solkI_N
    
    phi0 = np.interp(Ns_k, Ns, sol0[:,0])
    dphi0 = np.interp(Ns_k, Ns, sol0[:,1])

    return (3 / 8) ** 2 * k ** 3 / (2 * np.pi ** 2) * np.abs(delta(phi0[-1], dphi0[-1], solk_N[-1,0], solk_N[-1,1])) ** 2

invMpc_to_GeV = 6.3949e-39
km_per_sec = 3.3356e-6
H0 = 70 * km_per_sec * invMpc_to_GeV # in GeV
omegam = 0.32
aeq = 1 / 3400
gstar_s = 3.9
gstar_rho = 3.4
gstar_rh = 106.75
a_today = np.sqrt(HI / H0) / (omegam * aeq) ** (1 / 4) * gstar_rh ** (1 / 12) * gstar_rho ** (1 / 4) / gstar_s ** (1 / 3)
print('HI: {:.2e} GeV'.format(HI))
print('a_today: {:.2e}'.format(a_today))

ks_today = np.logspace(-4, 6, num = 100) # in Mpc^-1
ks_rh = ks_today * a_today * invMpc_to_GeV # in GeV
Pdel = np.array([delsqr(k) for k in ks_rh])
coeff = (3 / 8) ** 2 / 12 / np.pi ** 2 / alpha[-1] * dV(sol0[-1,0]) ** 2 * ddV(sol0[-1,0]) / V(sol0[-1,0]) ** 2
exp = np.interp(ks_rh, HI * np.exp(Ns), -2 * np.cumsum(alpha[::-1])[::-1] * (Ns[1] - Ns[0]))
np.savez(filename, HI = HI, ks = ks_today, Pdel = Pdel, coeff = coeff, exp = exp)
