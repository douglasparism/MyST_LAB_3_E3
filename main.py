
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import functions as fn
import mt5_main as mt5_lib

doug_acc_dic = {"local_exe":'C:\\Users\\DParis\\AppData\\Roaming\\XM Global MT5\\terminal64.exe'}
doug_acc_dic["mt5_acc"] = 5401675
doug_acc_dic["mt5_inv_pas"] = "vdGVQp8v"
df_prices_doug,df_hist_doug = mt5_lib.get_mt5_df(doug_acc_dic["local_exe"],doug_acc_dic["mt5_acc"]
                                                 ,doug_acc_dic["mt5_inv_pas"])
