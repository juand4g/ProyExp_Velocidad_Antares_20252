from Modelo import *
import pandas as pd

#Rutas de archivos
archivo_espectro = "Espectros limpios/20250809_antares.dat"
archivo_lampara = "Espectros limpios/20250809_thar.dat"
archivo_lineas_lamp = "lineas_lampara.xlsx"
archivo_lineas_espectro = "lineas_espectro.xlsx"

espectro = pd.read_csv(archivo_espectro, sep=" ", lineterminator="\n", names=["px","y"])
lampara = pd.read_csv(archivo_lampara, sep=" ", lineterminator="\n", names=["px","y"])
lineas_lampara = pd.read_excel(archivo_lineas_lamp).sort_values("px")
lineas_espectro = pd.read_excel(archivo_lineas_espectro).sort_values("obs")

espectro["y"], _ = aplicar_filtro_hampel(espectro, num_sigmas=2.5)

mostrar_espectro_y_lampara(espectro,lampara)

espectro["wl"] = ajustar_calibracion(espectro, lineas_lampara, mostrar=True, orden=1)

espectro["envolvente"] = calcular_envolvente_pchip(espectro["wl"], espectro["y"],mostrar=True, window_size=50)
espectro["y_norm"] = espectro["y"]/espectro["envolvente"]


mostrar_espectro_normalizado(espectro)

V, v_prom, inc = calcular_velocidades(lineas_espectro["obs"], lineas_espectro["nat"], mostrar=True)

v_prom, inc = v_prom/1e3, inc/1e3 #pasar a km

print(f"Velocidad radial promedio: ({v_prom:.7g} ± {inc:.1g}) km/s")