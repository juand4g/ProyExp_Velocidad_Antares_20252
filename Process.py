import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Hampel import hampel_filter
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

archivo_espectro = "Espectros limpios/20250809_antares.dat"
archivo_lampara = "Espectros limpios/20250809_thar.dat"

espectro = pd.read_csv(archivo_espectro, sep=" ", lineterminator="\n", names=["px","y"])
lampara = pd.read_csv(archivo_lampara, sep=" ", lineterminator="\n", names=["px","y"])

espectro["y"], marcados = hampel_filter(espectro["y"], window=15, n_sigmas=4, replace="interp")

def monstrar_espectro_y_lampara():
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

# (px, wl)
lineas_thar = [
    (922.51, 6577.2148),
    (995.00, 6583.9058),
    (1044.65, 6588.5396),
    (1076.504, 6591.4844),
    (1102.996, 6593.9390),
    (724.999, 6558.8755),
    (673.999, 6554.1602),
    (646.98, 6545.7188),
    (577.02, 6538.1118),
    (501.000, 6531.3418)
]

lineas_thar_array = np.array(lineas_thar)
# Ordenar por la primera columna (pixeles)
lineas_thar_array = lineas_thar_array[lineas_thar_array[:,0].argsort()]


# Separar en dos arrays
pixeles = lineas_thar_array[:,0]
longitudes = lineas_thar_array[:,1]

orden_ajuste = 3
coeficientes_calibracion = np.polyfit(pixeles, longitudes, orden_ajuste)

# Evaluar el polinomio en los pixeles
ajuste = np.polyval(coeficientes_calibracion, pixeles)

def mostrar_calibracion():
    plt.scatter(pixeles, longitudes, label="Líneas identificadas")
    plt.plot(pixeles, ajuste, c="r", label=f"Ajuste de orden {orden_ajuste}")
    plt.xlabel("Píxeles")
    plt.ylabel("Longitud de onda (Å)")
    plt.show()

espectro["wl"] = np.polyval(coeficientes_calibracion, espectro["px"])

def mostrar_espectro_calibrado():
    plt.plot(espectro["wl"], espectro["y"], color="tab:blue")
    plt.ylabel("Flujo estrella")
    plt.xlabel("Longitud de onda (Å)")
    plt.grid()
    plt.show()


mostrar_espectro_calibrado()

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

    envolvente_ajustada = envolvente.copy()
    envolvente_ajustada[0:2] = envolvente[2]
    envolvente_ajustada[-2:-1] = envolvente[-3]
    envolvente_ajustada[-1] = envolvente[-3]
    
    print(envolvente_ajustada[-10:])

    if mostrar:
        plt.plot(wavelength,flux,label="Espectro", color="blue")
        plt.plot(wavelength, envolvente_ajustada, label="Envolvente", color="red")
        plt.legend()
        plt.grid()
        plt.title("Cálculo de envolvente")
        plt.show()
    
    return envolvente_ajustada

espectro["envolvente"] = calcular_envolvente(espectro["wl"], espectro["y"],mostrar=True)
espectro["y_norm"] = espectro["y"]/espectro["envolvente"]

def mostrar_espectro_normalizado():
    plt.plot(espectro["wl"], espectro["y_norm"], color="tab:blue")
    plt.ylabel("Flujo estrella")
    plt.xlabel("Longitud de onda (Å)")
    plt.title("Espectro normalizado")
    plt.grid()
    plt.show()

mostrar_espectro_normalizado()
