import numpy as np

def hampel_filter(y, window=15, n_sigmas=5.0, replace='median'):
    """
    y: array 1D del espectro
    window: semiancho de la ventana (total=2*window+1)
    n_sigmas: umbral en unidades de sigma robusto (1.4826*MAD)
    replace: 'median' o 'interp' (lineal)
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    y_clean = y.copy()
    idx_out = np.zeros(n, dtype=bool)

    for i in range(n):
        i0 = max(0, i - window)
        i1 = min(n, i + window + 1)
        med = np.median(y[i0:i1])
        mad = np.median(np.abs(y[i0:i1] - med)) + 1e-12
        sigma = 1.4826 * mad
        if np.abs(y[i] - med) > n_sigmas * sigma:
            idx_out[i] = True

    if replace == 'median':
        for i in np.where(idx_out)[0]:
            i0 = max(0, i - window)
            i1 = min(n, i + window + 1)
            y_clean[i] = np.median(y[i0:i1])
    elif replace == 'interp':
        # reemplazo por interpolación lineal a través de los puntos buenos
        good = ~idx_out
        y_clean[idx_out] = np.interp(np.where(idx_out)[0], np.where(good)[0], y[good])

    return y_clean, idx_out
