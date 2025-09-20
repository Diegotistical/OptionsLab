import numpy as np

def black_scholes_price_delta_gamma(S, K, T, r, sigma, q):
    # Vectorized BS price, delta, gamma
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    from scipy.stats import norm
    price = np.exp(-q*T) * S * norm.cdf(d1) - np.exp(-r*T) * K * norm.cdf(d2)
    delta = np.exp(-q*T) * norm.cdf(d1)
    gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return price, delta, gamma

def geometric_asian_price_discrete(S, K, T, r, sigma, q, n_steps=12):
    t = np.linspace(T/n_steps, T, n_steps, axis=-1) if np.ndim(T) == 0 else np.linspace(T[:, None]/n_steps, T[:, None], n_steps, axis=1)
    w = np.ones(n_steps)/n_steps
    mu_ln = np.log(S) + np.sum((r - q - 0.5*sigma**2)[:, None] * t * w, axis=1)
    var_ln = np.sum((sigma[:, None]**2 * w**2 * t), axis=1)
    price = np.exp(-r*T) * (np.exp(mu_ln + 0.5*var_ln) - K)
    return price
