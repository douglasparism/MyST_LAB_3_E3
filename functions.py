"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# basic
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from typing import Dict
import sys
import yfinance
# data and processing
if sys.platform not in ["darwin", "linux"]:
    import MetaTrader5 as mt5


# -------------------------------------------------------------------------- MT5: INITIALIZATION / LOGIN -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_init_login(param_acc, param_pass, param_exe):
    """
    Initialize conexion and Login into a Meta Trader 5 account in the local computer where this code is executed,
    using the MetaTrader5 python package.

    Parameters
    ----------

    param_acc: int
        accout number used to login into MetaTrader5 Web/Desktop App (normally is a 8-9 digit integer number)
        param_acc = 41668916
    param_pass: str
        accout trader's password (or just password) to login into MetaTrader5 Web/Desktop App
        (normally alphanumeric include uppercase and sometimes symbols). If the investor's password
        is provided, the some actions do not work like open trades.
        param_pass = "n2eunlnt"

    param_exe: str
        Route in disk where is the executable file of the MetaTrader5 desktop app which will be used
        param_direxe = 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'

    Return
    ------

        if connection is succesful then returns connected client object and prints message,
        if connection is not succesful then returns error message and attempts a shutdown of connection.

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5login_py


    """

    # server name (as it is specified in the terminal)
    mt5_ser = "FxPro-MT5"
    # timeout (in miliseconds)
    mt5_tmo = 10000

    # Perform initialization handshake
    ini_message = mt5.initialize(param_exe, login=param_acc, password=param_pass, server=mt5_ser,
                                 timeout=mt5_tmo, portable=False)

    # resulting message
    if not ini_message:
        print(" **** init_login failed, error code =", mt5.last_error())
        mt5.shutdown()
    else:
        print(" ++++ init_login succeded, message = ", ini_message)

    # returns an instance of a connection object (or client)
    return mt5


# ------------------------------------------------------------------------------------ MT5: ACCOUNT INFO -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_acc_info(param_ct):
    """
    Get the info of the account associated with the initialized client param_ct

    Params
    ------

    param_ct: MetaTrader5 initialized client object
        this is an already succesfully initialized conexion object to MetaTrader5 Desktop App

    Returns
    -------

    df_acc_info: pd.DataFrame
        Pandas DataFrame with the account info

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5login_py

    """

    # get the account info and transform it into a dataframe format
    acc_info = param_ct.account_info()._asdict()

    # select especific info to display
    df_acc_info = pd.DataFrame(list(acc_info.items()), columns=['property', 'value'])

    # return dataframe with the account info
    return df_acc_info


# ------------------------------------------------------------------------------- MT5: HISTORICAL ORDERS -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_hist_trades(param_ct, param_ini, param_end):
    """
    Get the historical executed trades in the account associated with the initialized MetaTrader5 client

    Params
    ------

    param_ct: MetaTrader5 initialized client object
        This is an already succesfully initialized conexion object to MetaTrader5 Desktop App

    param_ini: datetime
        Initial date to draw the historical trades

        param_ini = datetime(2021, 2, 1)
    param_end: datetime
        Final date to draw the historical trades

        param_end = datetime(2021, 3, 1)

    Returns
    -------
        df_hist_trades: pd.DataFrame

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5historydealsget_py
    https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties

    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5historyordersget_py
    https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer

    """

    # get historical info of deals in the account
    history_deals = param_ct.history_deals_get(param_ini, param_end)

    # get historical info of orders in the account
    history_orders = param_ct.history_orders_get(param_ini, param_end)

    # check for returned results
    if (len(history_orders) > 0) & (len(history_deals) > 0):
        print(" ++++ Historical orders retrive: OK")
        print(" ++++ Historical deals retrive: OK")
    else:
        print("No orders and/or deals returned")

    # historical deals of the account
    df_deals = pd.DataFrame(list(history_deals), columns=history_deals[0]._asdict().keys())

    # historical orders of the account
    df_orders = pd.DataFrame(list(history_orders), columns=history_orders[0]._asdict().keys())

    # useful columns from orders
    df_hist_trades = df_orders[['time_setup', 'symbol', 'position_id', 'type', 'volume_current',
                                'price_open', 'sl', 'tp']]

    # useful columns from deals
    df_deals_hist = df_deals[['position_id', 'type', 'price', 'volume']]

    # rename columns
    df_hist_trades.columns = ['OpenTime', 'Symbol', 'Ticket', 'Type', 'Volume', 'OpenPrice', 'S/L', 'T/P']
    df_deals_hist.columns = ['Ticket', 'Type', 'Price', 'Volume']

    # choose only buy or sell transactions (ignore all the rest, like balance ...)
    df_hist_trades = df_hist_trades[(df_hist_trades['Type'] == 0) | (df_hist_trades['Type'] == 1)]
    df_deals_hist = df_deals_hist[(df_deals_hist['Type'] == 0) | (df_deals_hist['Type'] == 1)]
    df_hist_trades['OpenTime'] = pd.to_datetime(df_hist_trades['OpenTime'], unit='s')

    # unique values for position_id
    uni_id = df_hist_trades['Ticket'].unique()

    # first and last index for every unique value of position_id
    ind_profloss = [df_hist_trades.index[df_hist_trades['Ticket'] == i][0] for i in uni_id]
    ind_open = [df_deals_hist.index[df_deals_hist['Ticket'] == i][0] for i in uni_id]
    ind_close = [df_deals_hist.index[df_deals_hist['Ticket'] == i][-1] for i in uni_id]

    # generate lists with values to add
    cts = df_hist_trades['OpenTime'].loc[ind_open]
    ops = df_deals_hist['Price'].loc[ind_open]
    cps = df_deals_hist['Price'].loc[ind_close]
    vol = df_deals_hist['Volume'].loc[ind_close]
    # resize dataframe to have only the first value of every unique position_id
    df_hist_trades = df_hist_trades.loc[ind_profloss]

    # add close time and close price as a column to dataframe
    df_hist_trades['CloseTime'] = cts.to_list()
    df_hist_trades['OpenPrice'] = ops.to_list()
    df_hist_trades['ClosePrice'] = cps.to_list()
    df_hist_trades['Volume'] = vol.to_list()
    df_hist_trades['Profit'] = df_deals['profit'].loc[df_deals['position_id'].isin(uni_id) &
                                                      df_deals['entry'] == 1].to_list()
    return df_hist_trades


# ------------------------------------------------------------------------------- MT5: HISTORICAL PRICES -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_hist_prices(param_ct, param_sym, param_tf, param_ini, param_end):
    """
    Historical prices retrival from MetaTrader 5 Desktop App.

    Parameters
    ----------

    param_ct: MetaTrader5 initialized client object
        This is an already succesfully initialized conexion object to MetaTrader5 Desktop App

    param_sym: str
        The symbol of which the historical prices will be retrieved

        param_sym = 'EURUSD'

    param_tf: str
        The price granularity for the historical prices. Check available timeframes and nomenclatures from
        the references. The substring 'TIMEFRAME_' is automatically added.

        param_tf = 'M1'
    param_ini: datetime
        Initial date to draw the historical trades

        param_ini = datetime(2021, 2, 1)
    param_end: datetime
        Final date to draw the historical trades

        param_end = datetime(2021, 3, 1)

    **** WARNINGS ****

    1.- Available History

        MetaTrader 5 terminal provides bars only within a history available to a user on charts. The number of # bars available to users is set in the "Max.bars in chart" parameter. So this must be done
        manually within the desktop app to which the connection is made.

    2.- TimeZone
        When creating the 'datetime' object, Python uses the local time zone,
        MetaTrader 5 stores tick and bar open time in UTC time zone (without the shift).
        Data received from the MetaTrader 5 terminal has UTC time.
        Perform a validation whether if its necessary to shift time to local timezone.

    **** ******** ****

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5copyratesfrom_py#timeframe

    """

    # get hour info in UTC timezone (also GMT+0)
    hour_utc = datetime.datetime.now().utcnow().hour
    # get hour info in local timezone (your computer)
    hour_here = datetime.datetime.now().hour
    # difference (in hours) from UTC timezone
    diff_here_utc = hour_utc - hour_here
    # store the difference in hours
    tdelta = datetime.timedelta(hours=diff_here_utc)
    # granularity
    param_tf = getattr(param_ct, 'TIMEFRAME_' + param_tf)
    # dictionary for more than 1 symbol to retrieve prices
    d_prices = {}

    # retrieve prices for every symbol in the list
    for symbol in param_sym:
        # prices retrival from MetaTrader 5 Desktop App
        prices = pd.DataFrame(param_ct.copy_rates_range(symbol, param_tf,
                                                        param_ini - tdelta,
                                                        param_end - tdelta))
        # convert to datetime
        prices['time'] = [datetime.datetime.fromtimestamp(times) for times in prices['time']]

        # store in dict with symbol as a key
        d_prices[symbol] = prices

    # return historical prices
    return d_prices

# PARTE 1


def f_pip_size(params_ins: str) -> int:
    inst_pips = pd.read_csv('./files/instruments_pips.csv')
    inst_pips.Instrument = inst_pips.Instrument.map(lambda x: "".join(x.split("_")))
    ins = inst_pips[inst_pips['Instrument'] == params_ins]
    if len(ins) == 0:
        return 100
    return int(1 / float(ins['TickSize']))


def f_columnas_tiempos(param_data: pd.DataFrame) -> pd.DataFrame:
    param_data['tiempo'] = (param_data['CloseTime'] - param_data['OpenTime'])\
        .map(lambda x: x.total_seconds())
    return param_data


def f_columnas_pips(param_data: pd.DataFrame) -> pd.DataFrame:
    mults = param_data.loc[:, 'mult'].astype(int)
    close = param_data.loc[:, 'ClosePrice'].astype(float)
    opn = param_data.loc[:, 'OpenPrice'].astype(float)
    type = param_data.loc[:, 'Type']

    temp = pd.DataFrame({
        'type': type,
        'mult': mults,
        'open': opn,
        'close': close
    })

    pipSeries = temp.apply(
        lambda x: (x[3] - x[2]) * x[1] if x[0] == 'buy' # aqui debo poner 0?
        else (x[2] - x[3]) * x[1], 1
    )

    pips_acm = pipSeries.cumsum()
    profit_acm = param_data.loc[:, 'Profit'].cumsum()
    param_data['pips'] = pipSeries
    param_data['pips_acm'] = pips_acm
    param_data['profit_acm'] = profit_acm
    return param_data


def f_estadisticas_ba(param_data: pd.DataFrame) -> Dict:
    medida = [
        'Ops totales',
        'Ganadoras',
        'Ganadoras_c',
        'Ganadoras_v',
        'Perdedoras',
        'Perdedoras_c',
        'Perdedoras_v',
        'Mediana (Profit)',
        'Mediana (Pips)',
        'r_efectividad',
        'r_proporcion',
        'r_efectividad_c',
        'r_efectividad_v'
    ]

    desc = [
        'Operaciones totales',
        'Operaciones ganadoras',
        'Operaciones ganadoras de compra',
        'Operaciones ganadoras de venta',
        'Operaciones eprdedoras',
        'Operaciones perdedoras de compra',
        'Operaciones eprdedoras de venta',
        'Mediana de profit de operaciones',
        'Mediana de pips de operaciones',
        'Ganadoras Totales/Operaciones Totales',
        'Ganadoras Totales/Perdedoras Totales',
        'Ganadoras Compras/Operaciones Totales',
        'Ganadoras Ventas / Operaciones Totales'
    ]

    valor = [
        len(param_data),
        len(param_data[param_data['Profit'] > 0]),
        len(param_data[(param_data['Profit'] > 0) &
            (param_data['Type'] == '0')]),
        len(param_data[(param_data['Profit'] > 0) &
            (param_data['Type'] == '1')]),
        len(param_data[param_data['Profit'] < 0]),
        len(param_data[(param_data['Profit'] < 0) &
            (param_data['Type'] == '0')]),
        len(param_data[(param_data['Profit'] < 0) &
            (param_data['Type'] == '0')]),
        param_data['Profit'].mean(),
        param_data['pips'].mean(),
        len(param_data[param_data['Profit'] > 0]) / len(param_data),
        len(param_data[param_data['Profit'] > 0]) /
        len(param_data[param_data['Profit'] < 0]),
        len(param_data[(param_data['Profit'] > 0) &
            (param_data['Type'] == '0')]) / len(param_data),
        len(param_data[(param_data['Profit'] > 0) &
            (param_data['Type'] == '1')]) / len(param_data),
    ]

    df1 = pd.DataFrame(
        {
            'medida': medida,
            'valor': valor,
            'descripcion': desc
        }
    )

    tmp = param_data.loc[:, ['Symbol', 'Profit']]
    tmp["rank"] = tmp.Profit.map(lambda x: 1 if x > 0 else 0)
    df2 = tmp.loc[:, ['Symbol', 'rank']].groupby('Symbol').sum() /\
        tmp.loc[:, ['Symbol', 'rank']].groupby('Symbol').count()
    df2 = df2.reset_index()
    df2["rank"] = df2["rank"].map(lambda x: f'{x*100:.2f}%')

    return {
        'df_1_tabla': df1,
        'df_2_ranking': df2
    }


def f_evolucion_capital(cap_ini,operaciones):
    start_date = operaciones.OpenTime.min().replace(hour=23, minute=59, second=59)
    end_date = operaciones.CloseTime.max()
    timestamp = [start_date + timedelta(n) for n in range(int((end_date - start_date).days)+2)]

    profit_acm = [operaciones[operaciones.CloseTime <= i].Profit.sum() for i in timestamp]
    profit_acm_d = [i+cap_ini for i in profit_acm]
    profit_d = [0] + [profit_acm[i] - profit_acm[i - 1] for i in range(1, len(profit_acm))]
    profit_d[0] = profit_acm_d[0] - cap_ini

    parte_2_df = pd.DataFrame(timestamp)
    parte_2_df["profit_d"] = profit_d
    parte_2_df["profit_acm_d"] = profit_acm_d
    parte_2_df.rename(columns={parte_2_df.columns[0]: "timestamp"}, inplace=True)
    return parte_2_df


def SR_orig(parte_2, rf):
    rends = np.log(parte_2['profit_acm_d'] / parte_2['profit_acm_d'].shift(1)).dropna()
    rp = rends.mean()
    sdp = rends.std(ddof=1)*np.sqrt(252)
    return (rp-rf)/sdp


def SR_actualizado(parte_2, benchmark_ticker):
    bnchmrk = yfinance.download(benchmark_ticker, start=str(parte_2['timestamp'][0].date()),
                                end=str(parte_2['timestamp'][len(parte_2) - 1].date() + timedelta(days=1)))

    rends = np.log(parte_2['profit_acm_d'] / parte_2['profit_acm_d'].shift(1)).dropna()
    rends_bnchmrk = np.log(bnchmrk['Adj Close'] / bnchmrk['Adj Close'].shift(1)).dropna()
    dates = parte_2.iloc[1:, 0].map(lambda x: str(x).split(' ')[0]).values
    diff = rends.values.copy()
    for r in range(len(rends.values)):
        sp_dates = list(rends_bnchmrk.index.astype(str))
        try:
            i = sp_dates.index(dates[r])
        except ValueError:
            i = -1
        if i != -1:
            diff[r] = diff[r] - rends_bnchmrk.values[i]
    #diff = rends.values - rends_bnchmrk.values

    sdp = diff.std(ddof=1) * np.sqrt(252)
    r_trader = rends.mean() * 252
    r_benchmark = rends_bnchmrk.mean() * 252
    return (r_trader - r_benchmark) / sdp


def drawdown(parte_2):
    xs = parte_2['profit_acm_d']
    dw = np.maximum.accumulate(xs) - xs
    if np.sum(dw) > 0:
        i = np.argmax(dw)
        j = np.argmax(xs[:i])
        drawdown = (parte_2.loc[i]['profit_acm_d'] - parte_2.loc[j]['profit_acm_d']) / parte_2.loc[j]['profit_acm_d']

    else:
        i = 0
        j = 0
        drawdown = 0

    ini_date = parte_2.loc[j]['timestamp']
    end_date = parte_2.loc[i]['timestamp']

    capital = drawdown * parte_2.loc[j]['profit_acm_d']
    return drawdown, capital, ini_date, end_date


def drawup(parte_2):
    xs = parte_2['profit_acm_d']
    dw = np.maximum.accumulate(xs)
    if np.sum(dw) > 0:
        i = np.argmax(dw)
        j = np.argmin(xs[:i])  # start of period
        drawup = (parte_2.loc[i]['profit_acm_d'] - parte_2.loc[j]['profit_acm_d']) / parte_2.loc[j]['profit_acm_d']

    else:
        i = 0
        j = 0
        drawup = 0

    ini_date = parte_2.loc[j]['timestamp']
    end_date = parte_2.loc[i]['timestamp']

    capital = drawup * parte_2.loc[j]['profit_acm_d']
    return drawup, capital, ini_date, end_date


def f_estadisticas_mad(parte_2_1, rf, benchmark_ticker):
    metrica = ["sharpe_original", "sharpe_actualizado", "drawdown_capi", "drawdown_capi", "drawdown_capi",
               "drawup_capi", "drawup_capi", "drawup_capi"]
    formato = ['Cantidad', 'Cantidad', 'Fecha Inicial', 'Fecha Final', 'DrawDown $ (capital)', 'Fecha Inicial',
               'Fecha Final', 'DrawUp $ (capital)']
    descripcion = ['Sharpe Ratio Fórmula Original', 'Sharpe Ratio Fórmula Ajustada',
                   'Fecha inicial del DrawDown de Capital', 'Fecha final del DrawDown de Capital',
                   'Máxima pérdida flotante registrada', 'Fecha inicial del DrawUp de Capital',
                   'Fecha final del DrawUp de Capital', 'Máxima ganancia flotante registrada']
    _, capital_up, ini_date_up, end_date_up = drawup(parte_2_1)
    _, capital_down, ini_date_down, end_date_down = drawdown(parte_2_1)

    valor = [SR_orig(parte_2_1, rf), SR_actualizado(parte_2_1, benchmark_ticker),
             ini_date_down, end_date_down, capital_down, ini_date_up, end_date_up, capital_up]

    estadisticas_mad_df = pd.DataFrame(metrica)
    estadisticas_mad_df["formato"] = formato
    estadisticas_mad_df["valor"] = valor
    estadisticas_mad_df["descripcion"] = descripcion

    return estadisticas_mad_df


def get_mt5_prices(symbols, start_time, end_time,
                   local_exe = 'C:\\Users\\DParis\\AppData\\Roaming\\XM Global MT5\\terminal64.exe',
                   mt5_acc = 5401675, mt5_inv_pas = "vdGVQp8v"):

    mt5_client = f_init_login(param_acc=mt5_acc,
                              param_pass=mt5_inv_pas,
                              param_exe=local_exe)

    end_date = end_time + datetime.timedelta(hours=23)
    ini_date = start_time - datetime.timedelta(hours=23)

    # get historical prices using UTC time
    df_prices = f_hist_prices(param_ct=mt5_client,
                              param_sym=symbols,
                              param_tf='M1',
                              param_ini=ini_date, param_end=end_date)

    return df_prices


def f_be_de(param_data,local_exe = 'C:\\Users\\DParis\\AppData\\Roaming\\XM Global MT5\\terminal64.exe'):

    winners = param_data.copy()[param_data.Profit > 0]
    winners = winners.reset_index(drop=True)
    winners["Ratio"] = winners["Profit"] / abs(winners["profit_acm"])

    n_occ = []
    occu = []
    for i in winners.index:
        df = param_data.copy()[(param_data.OpenTime <= winners["CloseTime"][i]) &
                        (param_data.CloseTime > winners["CloseTime"][i])]

        df['CTime_Anchor'] = pd.Timestamp(winners['CloseTime'][i])
        occu.append(df)
        n_occ.append(len(df))
    occu_df = pd.concat(occu, ignore_index=True)
    prices_df = get_mt5_prices(symbols=occu_df.Symbol.tolist(),
                               start_time=occu_df.CTime_Anchor.min(),
                               end_time=occu_df.CTime_Anchor.max(),local_exe=local_exe)

    prices_filtered = [prices_df[occu_df.Symbol[i]][prices_df[occu_df.Symbol[i]].time == occu_df.CTime_Anchor[i].replace(second=0)]["close"].values for i in occu_df.index]
    if len(prices_filtered)<1:
        print("sin ocurrencias")
    else:
        occu_df["Price_CTime_Anchor"] = np.column_stack(prices_filtered).ravel()
        occu_df["P&L_CTime_Anchor"] = (occu_df.OpenPrice - occu_df.Price_CTime_Anchor) * occu_df.Volume

        final_dict = {'ocurrencias': {'Cantidad': len(occu_df)}}
        s_quo = []
        a_perdida = []
        for i in occu_df.index:
            inst = winners[winners['CloseTime'] == occu_df['CTime_Anchor'][i]]
            final_dict['ocurrencias'][f'ocurrencia_{i + 1}'] = {'timestamp': occu_df['CTime_Anchor'][i],
                                                                'operaciones': {
                                                                    'ganadoras': {'instrumento': inst['Symbol'].iloc[0],
                                                                                  'volumen': inst['Volume'].iloc[0],
                                                                                  'sentido': "buy" if inst['Type'].iloc[0] == 0 else "sell",
                                                                                  'profit_ganadora': round(
                                                                                      inst['Profit'].iloc[0],
                                                                                      2)},
                                                                    'perdedoras': {'instrumento': occu_df['Symbol'][i],
                                                                                   'volumen': occu_df['Volume'][i],
                                                                                   'sentido': "buy" if inst['Type'].iloc[0] == 0 else "sell",
                                                                                   'profit_perdedora': round(
                                                                                       occu_df['P&L_CTime_Anchor'][i], 2)}
                                                                },
                                                                'ratio_cp_profit_acm': round(
                                                                    abs(occu_df['P&L_CTime_Anchor'][i] /
                                                                        inst['profit_acm'].iloc[0]),
                                                                    2),
                                                                'ratio_cg_profit_acm': round(
                                                                    abs(inst['Profit'].iloc[0] / inst['profit_acm'].iloc[
                                                                        0]), 2),
                                                                'ratio_cp_cg': round(
                                                                    abs(occu_df['P&L_CTime_Anchor'][i] /
                                                                        inst['Profit'].iloc[0]), 2)
                                                                }

            if abs(occu_df['P&L_CTime_Anchor'][i] / inst['profit_acm'].iloc[0]) < abs(
                    inst['Profit'].iloc[0] / inst['profit_acm'].iloc[0]):
                s_quo.append(1)
            else:
                s_quo.append(0)

            if abs(occu_df['P&L_CTime_Anchor'][i] / inst['Profit'].iloc[0]) > 2:
                a_perdida.append(1)
            else:
                a_perdida.append(0)

        def cond1():
            return winners['profit_acm'].iloc[0] < winners['profit_acm'].iloc[-1]

        def cond2():
            return winners['Profit'].iloc[-1] > winners['Profit'].iloc[0] and occu_df['P&L_CTime_Anchor'].iloc[-1] > \
                   occu_df['P&L_CTime_Anchor'].iloc[0]

        def cond3():
            return final_dict['ocurrencias'][f'ocurrencia_{max(len(occu_df)-1,1)}']['ratio_cp_cg'] > 2

        if (cond1() and cond2()) or (cond3() and cond2()) or (cond3() and cond1()):
            sensibilidad_dec = 'Si'
        else:
            sensibilidad_dec = 'No'

        final_dict['resultados'] = {
            'dataframe': pd.DataFrame({'ocurrencias': len(occu_df),
                                       'status_quo': str(round(100 * np.array(s_quo).mean(), 2)) + '%',
                                       'aversion_perdida': str(round(100 * np.array(a_perdida).mean(), 2)) + '%',
                                       'sensibilidad_decreciente': [sensibilidad_dec]})
        }

        return final_dict
