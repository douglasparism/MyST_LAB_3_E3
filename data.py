
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
from typing import Dict
import sys
if sys.platform not in ["darwin", "linux"]:
    import mt5_main as mt5_lib


def equipo(integrante, lib=False):
    if integrante == 1:
        if lib:
            doug_acc_dic = {
                "local_exe": 'C:\\Users\\DParis\\AppData\\Roaming\\XM Global MT5\\terminal64.exe'
            }
            doug_acc_dic["mt5_acc"] = 5401675
            doug_acc_dic["mt5_inv_pas"] = "vdGVQp8v"
            return doug_acc_dic
        return "./files/df_hist_5401675.csv"
    elif integrante == 2:
        return "./files/History5400732.xlsx"
    elif integrante == 3:
        return "./files/History5401437.xlsx"


def f_leer_archivo(param_estudiante: int,
                   param_lib: bool = False,
                   param_cred: Dict = {},
                   param_archivo: str = "") -> pd.DataFrame:
    if param_lib:
        param_cred = equipo(param_estudiante, lib=True)
        df_prices, df_hist = mt5_lib.get_mt5_df(
            param_cred["local_exe"],
            param_cred["mt5_acc"],
            param_cred["mt5_inv_pas"]
        )
        return df_hist
    param_archivo = equipo(param_estudiante)
    ext = param_archivo.split('.')[-1]
    if ext == 'csv':
        tem = pd.read_csv(param_archivo)
    elif ext in ['xls', 'xlsx']:
        tem = pd.read_excel(param_archivo)
    else:
        raise TypeError('Valid file types are `csv`, `xls` or `xlsx`.')
    tem['CloseTime'] = pd.to_datetime(tem['CloseTime'])
    tem['OpenTime'] = pd.to_datetime(tem['OpenTime'])
    return tem

    
