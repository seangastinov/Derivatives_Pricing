import numpy as np
from scipy.stats import qmc, norm

class Heston:
    def __init__(self, S0, T, r, q, mu, theta=0, sigma = 0, rho=0, kappa=0, V0=0, barrier=1743.525, price_history_matrix=None):
        self.S0 = S0
        self.barrier = barrier
        self.T = T
        self.r = r
        self.sigma = sigma
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.V0 = V0
        self.price_history_matrix = price_history_matrix
        self.mu = mu
        self.value_results = None
        self.q = q

    # Euler Discretization
    def simulate(self, n_trials, n_steps, antitheticVariates=False):

        # dt = self.T / n_steps
        dt = 1 / 252
        r = self.r

        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.V0

        self.n_trials = n_trials
        self.n_steps = n_steps


        cov = np.array([[1,rho],[rho,1]])
        Z = np.random.multivariate_normal(np.array([0,0]), cov, size=(n_trials,n_steps)) * np.sqrt(dt)

        if (antitheticVariates == True):
            n_trials *= 2
            self.n_trials = n_trials
            Z = np.concatenate((Z, -Z), axis=0)

        V_matrix = np.zeros((n_trials, n_steps + 1))
        V_matrix[:, 0] = v0
        s_matrix = np.zeros((n_trials, n_steps + 1))
        s_matrix[:, 0] = self.S0

        for j in range(self.n_steps):
            V_matrix[:, j + 1] = V_matrix[:, j] + kappa * (theta-V_matrix[:,j]) * dt + sigma * np.sqrt(V_matrix[:, j]) * Z[:, j, 1]
            V_matrix[:, j + 1] = np.maximum(V_matrix[:, j + 1], 0)
            s_matrix[:, j + 1] = s_matrix[:, j] * np.exp( (r - 0.5*V_matrix[:, j]) * dt + np.sqrt(V_matrix[:, j]) * Z[:, j, 0] )


        self.price_matrix = s_matrix
        return s_matrix
    


    def Pricer(self, price_matrix, S0 = 3487.05):
        # Assume buy 1 certificate
        n_trials = price_matrix.shape[0]
        barrier = self.barrier
        price_history_matrix = self.price_history_matrix
        redemptions = np.zeros(n_trials)
        barrier_hit = np.zeros(n_trials, dtype=bool)

        if np.any(price_history_matrix <= barrier):
            barrier_hit[:] = True


        for i in range(n_trials):
            S_T = price_matrix[i, -1]
            barrier_hit[i] = np.any(price_matrix[i, :] <= barrier)
            if not barrier_hit[i]:
                redemptions[i] = max(1000, 1000 * (1 + 1.5 * (S_T - S0) / S0))
            elif barrier_hit[i]:
                redemptions[i] = max(0, 1000 * (1 + (S_T - S0) / S0))



        avg_redemptions = np.mean(redemptions)
        std_redemptions = np.std(redemptions)
        redemptions_ROI = 1 + (avg_redemptions - 1000) / 1000
        return avg_redemptions, std_redemptions, redemptions_ROI
    
  

    def CVPricer(self, price_matrix, S0 = 3487.05):
        # Assume buy 1 certificate
        n_trials = price_matrix.shape[0]
        barrier = self.barrier
        price_history_matrix = self.price_history_matrix
        redemptions = np.zeros(n_trials)
        barrier_hit = np.zeros(n_trials, dtype=bool)

        # Proportion to Calculate c* and generate Xcv is 1:1000
        N_1 = n_trials//1001
        N_2 = n_trials - N_1
        control_redemptions = np.zeros(N_1)
        redemptions = np.zeros(N_2)
        barrier_hit_control = np.zeros(N_1, dtype=bool)
        barrier_hit = np.zeros(N_2, dtype=bool)

        if np.any(price_history_matrix <= barrier):
            barrier_hit_control[:] = True
            barrier_hit[:] = True
            print('Barrier Hit History')


        for i in range(N_1):
            S_T = price_matrix[i, -1]
            barrier_hit[i] = np.any(price_matrix[i, :] <= barrier)
            if not barrier_hit[i]:
                control_redemptions[i] = max(1000, 1000 * (1 + 1.5 * (S_T - S0) / S0))
            elif barrier_hit[i]:
                control_redemptions[i] = max(0, 1000 * (1 + (S_T - S0) / S0))

                
        cov_control_S = np.cov(control_redemptions, price_matrix[:N_1, -1])

        c_star = -cov_control_S[0,1] / cov_control_S[1,1]

        for i in range(N_2):
            S_T = price_matrix[i + N_1, -1]
            
            barrier_hit[i] = np.any(price_matrix[i, :] <= barrier)
            if not barrier_hit[i]:
                redemptions[i] = max(1000, 1000 * (1 + 1.5 * (S_T - S0) / S0)) + c_star * (S_T - np.mean(price_matrix[:N_1, -1]))
                redemptions[i] = max(1000, redemptions[i])
            elif barrier_hit[i]:
                redemptions[i] = max(0, 1000 * (1 + (S_T - S0) / S0)) + c_star * (S_T - np.mean(price_matrix[:N_1, -1]))
                redemptions[i] = max(0, redemptions[i])

        avg_redemptions = np.mean(redemptions)
        std_redemptions = np.std(redemptions)
        redemptions_ROI = 1 + (avg_redemptions - 1000) / 1000
        return avg_redemptions, std_redemptions, redemptions_ROI


    def QE_simulate(self, n_trials, n_steps, psi_c = 2, gamma1 = 0.5, gamma2 = 0.5, martingale_correction = True):

        # dt = self.T / n_steps
        dt = 1 / 252
        r = self.r
        q = self.q
        s0 = self.S0
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.V0

        self.n_trials = n_trials
        self.n_steps = n_steps

        sobol_gen = qmc.Sobol(d=n_steps + 1, scramble=True)
        Uv = sobol_gen.random(n=n_trials).T
        Us = sobol_gen.random(n=n_trials).T
        Zv, Zs = norm.ppf(Uv), norm.ppf(Us)

        
        E = np.exp(-kappa * dt)
        K0 = -(kappa * rho * theta) * dt / sigma
        K1 = (kappa * rho / sigma - 1 / 2) * gamma1 * dt - rho / sigma
        K2 = (kappa * rho / sigma - 1 / 2) * gamma2 * dt + rho / sigma
        K3 = (1 - rho ** 2) * gamma1 * dt
        K4 = (1 - rho ** 2) * gamma2 * dt        
        A = K2 + 0.5 * K4

        if martingale_correction: 
            K0_star = np.empty(n_trials)
        
        S = np.zeros((n_steps+1, n_trials))
        V = np.zeros((n_steps+1, n_trials))

        S[0] = np.log(s0)
        V[0] = v0

        for t in range(1, n_steps+1):
            m = theta + (V[t - 1] - theta) * E 
            s2 = (V[t - 1] * sigma**2 * E)/kappa * (1 - E) + (theta * sigma**2)/(2 * kappa)*(1 - E)**2 
            
            # avoid division by zero
            m[m == 0] = 1e-100

            psi = s2 / m ** 2


            idx = psi <= psi_c
            # When psi <= psiC
            b = np.sqrt(2/psi[idx] - 1 + np.sqrt(2/psi[idx] * (2/psi[idx] - 1)))
            a = m[idx] / (1 + b**2)     
            V[t, idx] = a * (b + Zv[t, idx]) ** 2

            # When psi > psiC
            p = (psi[~idx] - 1) / (psi[~idx] + 1)
            beta = (1 - p) / m[~idx]

            idx2 = (Uv[t, ~idx] < p)
            V[t, ~idx] = np.where(idx2, 0, (1 / beta) * np.log((1 - p) / (1 - Uv[t, ~idx])))

            if martingale_correction:
                K0_star[idx]  = - (A * b**2 * a) / (1 - 2 * A * a) + 0.5 * np.log(1 - 2 * A * a) - (K1 + 0.5 * K3) * V[t-1, idx]
                K0_star[~idx] = - np.log(p + (beta * (1 - p)) / (beta - A)) - (K1 + 0.5 * K3) * V[t-1, ~idx]

                S[t] = S[t - 1] + K0_star + K1 * V[t - 1] + K2 * V[t] + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Zs[t] 
            else:
                S[t] = S[t - 1] + K0      + K1 * V[t - 1] + K2 * V[t] + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Zs[t] 

            nan_mask_S = np.isnan(S[t])
            nan_mask_V = np.isnan(V[t])
            if  nan_mask_S.any():
                n_nan = nan_mask_S.sum()
                S_t_expectation = np.nanmean(S[t])  # Use nanmean to ignore NaNs
                print(f"Warning: {n_nan} paths have NaN values at step {t} and are replaced with E[S_t]={S_t_expectation:.4f}")
                S[t][nan_mask_S] = S_t_expectation 
            if nan_mask_V.any():
                n_nan = nan_mask_V.sum()
                V_t_expectation = np.nanmean(V[t]) 
                print(f"Warning: {n_nan} paths have NaN values at step {t} and are replaced with E[V_t]={V_t_expectation:.4f}")
                V[t][nan_mask_V] = V_t_expectation

        price_matrix = np.exp(S).T
        V_matrix = V.T

        return price_matrix

    
    def QE_simulate_AV(self, n_trials, n_steps, psi_c = 2, gamma1 = 0.5, gamma2 = 0.5, martingale_correction = True):

        # dt = self.T / n_steps
        dt = 1 / 252
        r = self.r
        s0 = self.S0
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.V0
        q = self.q

        self.n_trials = n_trials
        self.n_steps = n_steps

        # Generate Sobol sequences
        sobol_gen = qmc.Sobol(d=2 * (n_steps + 1), scramble=True)
        U = sobol_gen.random(n=n_trials // 2).T  # Generate half the paths
        U_antithetic = 1 - U  # Antithetic counterparts
        U_combined = np.concatenate((U, U_antithetic), axis=1)

        # Split the combined U into Uv and Us for V and S
        Uv = U_combined[:n_steps + 1]
        Us = U_combined[n_steps + 1:]

        # Convert uniform random numbers to normal variates
        Zv, Zs = norm.ppf(Uv), norm.ppf(Us)
        
        E = np.exp(-kappa * dt)
        K0 = -(kappa * rho * theta) * dt / sigma
        K1 = (kappa * rho / sigma - 1 / 2) * gamma1 * dt - rho / sigma
        K2 = (kappa * rho / sigma - 1 / 2) * gamma2 * dt + rho / sigma
        K3 = (1 - rho ** 2) * gamma1 * dt
        K4 = (1 - rho ** 2) * gamma2 * dt        
        A = K2 + 0.5 * K4

        if martingale_correction: 
            K0_star = np.empty(n_trials)
        
        S = np.zeros((n_steps+1, n_trials))
        V = np.zeros((n_steps+1, n_trials))

        S[0] = np.log(s0)
        V[0] = v0

        for t in range(1, n_steps+1):
            m = theta + (V[t - 1] - theta) * E 
            s2 = (V[t - 1] * sigma**2 * E)/kappa * (1 - E) + (theta * sigma**2)/(2 * kappa)*(1 - E)**2 
            
            # avoid division by zero
            m[m == 0] = 1e-100

            psi = s2 / m ** 2


            idx = psi <= psi_c
            # When psi <= psiC
            b = np.sqrt(2/psi[idx] - 1 + np.sqrt(2/psi[idx] * (2/psi[idx] - 1)))
            a = m[idx] / (1 + b**2)     
            V[t, idx] = a * (b + Zv[t, idx]) ** 2

            # When psi > psiC
            p = (psi[~idx] - 1) / (psi[~idx] + 1)
            beta = (1 - p) / m[~idx]

            idx2 = (Uv[t, ~idx] < p)
            V[t, ~idx] = np.where(idx2, 0, (1 / beta) * np.log((1 - p) / (1 - Uv[t, ~idx])))

            if martingale_correction:
                K0_star[idx]  = - (A * b**2 * a) / (1 - 2 * A * a) + 0.5 * np.log(1 - 2 * A * a) - (K1 + 0.5 * K3) * V[t-1, idx]
                K0_star[~idx] = - np.log(p + (beta * (1 - p)) / (beta - A)) - (K1 + 0.5 * K3) * V[t-1, ~idx]

                S[t] = S[t - 1] + K0_star + K1 * V[t - 1] + K2 * V[t] + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Zs[t] 
            else:
                S[t] = S[t - 1] + K0      + K1 * V[t - 1] + K2 * V[t] + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Zs[t] 

            nan_mask_S = np.isnan(S[t])
            nan_mask_V = np.isnan(V[t])
            if  nan_mask_S.any():
                n_nan = nan_mask_S.sum()
                S_t_expectation = np.nanmean(S[t])  # Use nanmean to ignore NaNs
                print(f"Warning: {n_nan} paths have NaN values at step {t} and are replaced with E[S_t]={S_t_expectation:.4f}")
                S[t][nan_mask_S] = S_t_expectation 
            if nan_mask_V.any():
                n_nan = nan_mask_V.sum()
                V_t_expectation = np.nanmean(V[t]) 
                print(f"Warning: {n_nan} paths have NaN values at step {t} and are replaced with E[V_t]={V_t_expectation:.4f}")
                V[t][nan_mask_V] = V_t_expectation
                
                
        price_matrix = np.exp(S).T
        V_matrix = V.T

        return price_matrix

    
# import yfinance as yf
# from datetime import date

# underlying_ticker = yf.Ticker("^STOXX50E")
# start_date = '2024-08-01'

# end_date = '2024-10-31'
# price_history_matrix = underlying_ticker.history(period="1y", start = '2022-07-12', end = end_date)
# price_history = price_history_matrix['Close'].values


# heston = Heston(S0 = 4871, T = 255/252, r = 3.163/100, q = 0, mu = 7.41/100, theta = 4.10984301e-06, sigma = 1.00063686e-01, rho = -5.76484809e-01, kappa = 5.00413684e-03, V0 = yoopii   o, barrier = 1743.525, price_history_matrix = price_history)
# price_matrix_QE = heston.QE_simulate(n_trials=2**20, n_steps=2, martingale_correction=True)
# price_matrix_QE_AV = heston.QE_simulate_AV(n_trials=2**20, n_steps=255, martingale_correction=True)

# pricer = heston.Pricer(price_matrix_QE)
# pricer_CV = heston.CVPricer(price_matrix_QE)
# pricer_AV = heston.Pricer(price_matrix_QE_AV)
# pricer_CV_AV = heston.CVPricer(price_matrix_QE_AV)

# print('QE: ',np.mean(price_matrix_QE[:,-1]))
# print('QE_av', np.mean(price_matrix_QE_AV[:,62]))
# print('Pricer: ', pricer)
# print('Pricer_CV: ', pricer_CV)
# print('Pricer_AV: ', pricer_AV)
# print('Pricer_CV_AV: ', pricer_CV_AV)
