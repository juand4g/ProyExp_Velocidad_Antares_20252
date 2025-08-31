import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Hampel import hampel_filter
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from specutils.fitting import find_lines_derivative
from specutils import Spectrum
import astropy.units as u

#Constantes
c = 3e8 #m/s

def aplicar_filtro_hampel(espectro, ventana=15, num_sigmas=4, reemplazo="interp"):
    return hampel_filter(espectro["y"], window=ventana, n_sigmas=num_sigmas, replace=reemplazo)

def mostrar_espectro_y_lampara(espectro, lampara):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))

    # Primer recuadro: espectro
    ax1.plot(espectro["px"], espectro["y"], color="tab:blue")
    ax1.set_ylabel("Flujo estrella")
    ax1.set_title("Espectro estelar y lámpara de calibración")
    ax1.grid()

    # Segundo recuadro: lámpara
    ax2.plot(lampara["px"], lampara["y"], color="tab:orange")
    ax2.set_ylabel("Intensidad lámpara")
    ax2.set_xlabel("Píxeles")
    ax2.grid()

    plt.tight_layout()
    plt.show()

def ajustar_calibracion(espectro, lineas_calibracion, orden=3, mostrar=False):
    # Separar en dos arrays
    pixeles = lineas_calibracion["px"]
    longitudes = lineas_calibracion["wl"]

    coeficientes_calibracion = np.polyfit(pixeles, longitudes, orden)

    # Evaluar el polinomio en los pixeles
    ajuste = np.polyval(coeficientes_calibracion, pixeles)

    if mostrar:
        _mostrar_calibracion(pixeles, longitudes, ajuste, orden)
    
    return np.polyval(coeficientes_calibracion, espectro["px"])

def _mostrar_calibracion(pixeles, longitudes, ajuste, orden):
    plt.scatter(pixeles, longitudes, label="Líneas identificadas")
    plt.plot(pixeles, ajuste, c="r", label=f"Ajuste de orden {orden}")
    plt.xlabel("Píxeles")
    plt.title("Calibración")
    plt.ylabel("Longitud de onda (Å)")
    plt.show()

def mostrar_espectro_calibrado(espectro):
    plt.plot(espectro["wl"], espectro["y"], color="tab:blue")
    plt.ylabel("Flujo estrella")
    plt.xlabel("Longitud de onda (Å)")
    plt.title("Espectro calibrado")
    plt.grid()
    plt.show()

def calcular_envolvente(wavelength, flux, window_size=50, mostrar=False):
    """
    Calcula la envolvente superior del espectro encontrando máximos locales
    en ventanas de tamaño fijo y luego interpolando.
    """
    # Encontrar índices de máximos locales
    peaks, _ = find_peaks(flux, distance=window_size)
    
    # Asegurarse de incluir los extremos
    if 0 not in peaks:
        peaks = np.insert(peaks, 0, 0)
    if len(flux)-1 not in peaks:
        peaks = np.append(peaks, len(flux)-1)
    
    # Interpolar los máximos para crear la envolvente
    env_interp = interp1d(wavelength[peaks], flux[peaks], 
                         kind='cubic', 
                         fill_value='extrapolate')
    
    envolvente = env_interp(wavelength)

    # Suavizar opcionalmente la envolvente
    envolvente = np.convolve(envolvente, np.ones(5)/5, mode='same')

    if mostrar:
        plt.plot(wavelength,flux,label="Espectro", color="blue")
        plt.plot(wavelength, envolvente, label="Envolvente", color="red")
        plt.legend()
        plt.grid()
        plt.title("Cálculo de envolvente")
        plt.show()
    
    return envolvente

def calcular_envolvente_pchip(wavelength, flux, window_size=50, mostrar=False):
    peaks, _ = find_peaks(flux, distance=window_size)
    # Preferir picos verdaderos: excluir extremos si son valles
    if len(peaks) == 0:
        # fallback: uso mediana móvil como envolvente
        return uniform_filter1d(flux, size=101)

    # incluir extremos solo si son razonables máximos locales:
    if peaks[0] != 0 and flux[0] > np.median(flux[:window_size]):
        peaks = np.insert(peaks, 0, 0)
    if peaks[-1] != len(flux)-1 and flux.iloc[-1] > np.median(flux.iloc[-window_size:]):
        peaks = np.append(peaks, len(flux)-1)

    x_peaks = wavelength[peaks]
    y_peaks = flux[peaks]

    # Usar PCHIP para evitar overshoot
    interp = PchipInterpolator(x_peaks, y_peaks, extrapolate=True)
    envolvente = interp(wavelength)

    # Suavizado ligero
    envolvente = uniform_filter1d(envolvente, size=5)

    if mostrar:
        import matplotlib.pyplot as plt
        plt.plot(wavelength, flux, label='Espectro')
        plt.plot(wavelength, envolvente, label='Envolvente (PCHIP)', color='red')
        plt.legend(); plt.grid(); plt.show()

    return envolvente

def mostrar_espectro_normalizado(espectro):
    plt.plot(espectro["wl"], espectro["y_norm"], color="tab:blue")
    plt.ylabel("Flujo estrella")
    plt.xlabel("Longitud de onda (Å)")
    plt.title("Espectro normalizado")
    plt.grid()
    plt.show()

def graficar_velocidades(WL, V):
    plt.scatter(WL, V)
    plt.title("Velocidades radiales calculadas")
    plt.xlabel("Longitud de onda (Å)")
    plt.ylabel("Velocidad radial (m/s)")
    plt.grid()
    plt.axhline(0)
    plt.show()


def calcular_velocidades(obs_wl, nat_wl, mostrar=False):
    fraccion = obs_wl/nat_wl
    v = c*(fraccion-1)
    
    if mostrar:
        graficar_velocidades(obs_wl,v)

    prom = np.mean(v)
    inc = np.std(v)

    return v, prom, inc 

def obtener_posibles_lineas(espectro, flujo_minimo=-0.3):

    # Convierte a arrays numpy
    wavelength = espectro["wl"].to_numpy() * u.AA
    flux = (1-espectro["y"]).to_numpy() * u.adu   # puedes definir adu como unidad

    spectrum = Spectrum(spectral_axis=wavelength, flux=flux)

    tabla_lineas = find_lines_derivative(spectrum, flux_threshold=flujo_minimo)
    return tabla_lineas

