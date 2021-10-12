
"""
# -- -------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                          -- #
# -- script: main.py : python script with the main functionality          -- #
# -- author: YOUR GITHUB USER NAME                                        -- #
# -- license: GPL-3.0 License                                             -- #
# -- repository: YOUR REPOSITORY URL                                      -- #
# -- -------------------------------------------------------------------- -- #
"""
import functions as fn
import sys
import data as dt

if sys.platform not in ["linux", "darwin"]:
    data_ini_1 = dt.f_leer_archivo(param_estudiante=1, param_lib=True)
else:
    data_ini_1 = dt.f_leer_archivo(param_estudiante=1)

data_ini_2 = dt.f_leer_archivo(param_estudiante=2)
data_ini_3 = dt.f_leer_archivo(param_estudiante=3)


# DATOS DOUGLAS (Estudiante 2)
data_ap = fn.f_columnas_tiempos(data_ini_1)
data_ap["mult"] = data_ap.Symbol.map(lambda x: fn.f_pip_size(x))
data_ap = fn.f_columnas_pips(data_ap)
parte_1 = fn.f_estadisticas_ba(data_ap)

parte_2_1 = fn.f_evolucion_capital(cap_ini=100000, operaciones=data_ap.copy())

parte_2_2 = fn.f_estadisticas_mad(parte_2_1, rf=0.05, benchmark_ticker="^GSPC")

# DATOS ANA PAULA (Estudiante 2)
data_ap = fn.f_columnas_tiempos(data_ini_2)
data_ap["mult"] = data_ap.Symbol.map(lambda x: fn.f_pip_size(x))
data_ap = fn.f_columnas_pips(data_ap)
parte_1 = fn.f_estadisticas_ba(data_ap)

parte_2_1 = fn.f_evolucion_capital(cap_ini=100000, operaciones=data_ap.copy())

parte_2_2 = fn.f_estadisticas_mad(parte_2_1, rf=0.05, benchmark_ticker="^GSPC")

# DATOS JUAN PABLO (Estudiante 3)
data_ap = fn.f_columnas_tiempos(data_ini_3)
data_ap["mult"] = data_ap.Symbol.map(lambda x: fn.f_pip_size(x))
data_ap = fn.f_columnas_pips(data_ap)
parte_1 = fn.f_estadisticas_ba(data_ap)

parte_2_1 = fn.f_evolucion_capital(cap_ini=100000, operaciones=data_ap.copy())

parte_2_2 = fn.f_estadisticas_mad(parte_2_1, rf=0.05, benchmark_ticker="^GSPC")
