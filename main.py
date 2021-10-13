
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
import datetime

#if sys.platform not in ["linux", "darwin"]:
#    data_ini_1 = dt.f_leer_archivo(param_estudiante=1, param_lib=True)
#else:
data_ini_1 = dt.f_leer_archivo(param_estudiante=1)
data_ini_2 = dt.f_leer_archivo(param_estudiante=2)
data_ini_3 = dt.f_leer_archivo(param_estudiante=3)


# DATOS DOUGLAS (Estudiante 2)
data_dou = fn.f_columnas_tiempos(data_ini_3)
data_dou["mult"] = data_dou.Symbol.map(lambda x: fn.f_pip_size(x))
data_dou = fn.f_columnas_pips(data_dou)
parte_1_dou = fn.f_estadisticas_ba(data_dou)

parte_2_1_dou = fn.f_evolucion_capital(cap_ini=100000, operaciones=data_dou.copy())
final_dict = fn.f_be_de(data_dou,local_exe = 'C:\\Program Files\\MetaTrader 5\\terminal64.exe')
parte_2_2_dou = fn.f_estadisticas_mad(parte_2_1_dou, rf=0.05, benchmark_ticker="^GSPC")



# DATOS ANA PAULA (Estudiante 2)
data_ap = fn.f_columnas_tiempos(data_ini_2)
data_ap["mult"] = data_ap.Symbol.map(lambda x: fn.f_pip_size(x))
data_ap = fn.f_columnas_pips(data_ap)
parte_1_ap = fn.f_estadisticas_ba(data_ap)

parte_2_1_ap = fn.f_evolucion_capital(cap_ini=100000, operaciones=data_ap.copy())

parte_2_2_ap = fn.f_estadisticas_mad(parte_2_1_ap, rf=0.05, benchmark_ticker="^GSPC")


# DATOS JUAN PABLO (Estudiante 3)
data_jp = fn.f_columnas_tiempos(data_ini_3)
data_jp["mult"] = data_jp.Symbol.map(lambda x: fn.f_pip_size(x))
data_jp = fn.f_columnas_pips(data_jp)
parte_1_jp = fn.f_estadisticas_ba(data_jp)

parte_2_1_jp = fn.f_evolucion_capital(cap_ini=100000, operaciones=data_jp.copy())

parte_2_2_jp = fn.f_estadisticas_mad(parte_2_1_jp, rf=0.05, benchmark_ticker="^GSPC")


# CARGAR PRECIOS HISTÓRICOS DE MT5
unique_symbols = list(set(list(data_ini_1.loc[:, 'Symbol'].values) +
                      list(data_ini_2.loc[:, 'Symbol'].values) +
                      list(data_ini_3.loc[:, 'Symbol'].values)))


act_credentials = dt.equipo(1, lib=True)


