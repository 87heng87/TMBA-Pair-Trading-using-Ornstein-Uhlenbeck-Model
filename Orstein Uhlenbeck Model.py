import io
import pandas as pd
from datetime import datetime
import numpy as np
import requests
import matplotlib.pyplot as plt
import scipy.optimize as solver
from scipy.stats import linregress
from mpmath import mp
from mpmath import nsum, inf

plt.style.use('ggplot')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def YahooData(ticker, start, end):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
    }

    url = "https://query1.finance.yahoo.com/v7/finance/download/" + str(ticker)
    x = int(datetime.strptime(start, '%Y-%m-%d').strftime("%s"))
    y = int(datetime.strptime(end, '%Y-%m-%d').strftime("%s"))
    url += "?period1=" + str(x) + "&period2=" + str(y) + "&interval=1d&events=history&includeAdjustedClose=true"

    r = requests.get(url, headers=headers)
    pad = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)

    return pad


def co_integration(Pt, Qt):
    Pt = Pt.ffill()
    Qt = Qt.ffill()
    # x = Qt.pct_change().dropna()
    # y = Pt.pct_change().dropna()
    x = np.log(Qt) - np.log(Qt[0])
    y = np.log(Pt) - np.log(Pt[0])
    return linregress(x, y)[0]


def Xt(Pt, Qt, B):
    return np.log(Pt) - B * np.log(Qt)


# E[T1] 與 E[T2] 分成兩個函式算
def first_passage_time_one_sided_boundary(a, b):
    A = np.sqrt(2) * abs(a)
    B = np.sqrt(2) * abs(b)
    a_to_o = 0.5 * nsum(lambda k: ((-1) ** (k+1)) * ((A ** k)/mp.fac(k)) * mp.gamma((k/2)), [1, inf])
    if (a / b) > 0:
        b_to_o = 0.5 * nsum(lambda k: ((-1) ** (k+1)) * ((B ** k)/mp.fac(k)) * mp.gamma((k/2)), [1, inf])
        return a_to_o - b_to_o
    elif (a / b) < 0:
        o_to_b = 0.5 * nsum(lambda k: ((B ** k)/mp.fac(k)) * mp.gamma((k/2)), [1, inf])
        return a_to_o + o_to_b


def first_passage_time_two_sided_boundary(a, b):
    A = np.sqrt(2) * a
    B = np.sqrt(2) * b
    return 0.5 * nsum(lambda n: ((A ** (2 * n) - B ** (2 * n)) / mp.fac(2 * n)) * (mp.gamma(n)), [1, inf])


def object_function(x):
    a, b, c = x[0], x[1], x[2]
    numerator = a - b - c
    denominator = first_passage_time_one_sided_boundary(a, b) + first_passage_time_two_sided_boundary(a, b)
    return - numerator / denominator


def optimize(c):
    X = np.array([1, -1, c])
    con1 = {'type': 'ineq', 'fun': lambda x: x[0] - x[2]}
    con2 = {'type': 'eq', 'fun': lambda x: x[2] - c}
    con3 = {'type': 'ineq', 'fun': lambda x: x[0] + x[1]}
    maximize = solver.minimize(object_function, x0=X, constraints=[con1, con2, con3])
    return maximize


def _compute_log_likelihood(params: tuple, *args: tuple) -> float:
    # Setting given parameters
    theta, mu, sigma = params
    X, dt = args
    n = len(X)

    # Calculating log likelihood
    sigma_tilde_squared = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)

    summation_term = sum((X[1:] - X[:-1] * np.exp(-theta * dt) - mu * (1 - np.exp(-theta * dt))) ** 2)

    summation_term = -summation_term / (2 * n * sigma_tilde_squared)

    log_likelihood = (-np.log(2 * np.pi) / 2) + (-np.log(np.sqrt(sigma_tilde_squared))) + summation_term

    return -log_likelihood


def MLH(X_t):
    initial_guess = np.array([np.std(X_t) / np.mean(X_t), np.mean(X_t), np.std(X_t)])
    bounds = ((1e-5, None), (None, None), (1e-5, None))
    test = solver.minimize(_compute_log_likelihood, initial_guess, args=(X_t.values, 1), bounds=bounds)
    return test


def threshold_finder(xt, cost):
    parameters_of_MLH = MLH(xt).x
    theta, mu, sigma = parameters_of_MLH[0], parameters_of_MLH[1], parameters_of_MLH[2]
    print('theta=', theta, ', mu=', mu, ', sigma=', sigma)
    dimensionless_cost = cost * (np.sqrt(2 * theta) / sigma)
    ths = optimize(dimensionless_cost).x
    a = ths[0] * (sigma / np.sqrt(2 * theta)) + mu
    b = ths[1] * (sigma / np.sqrt(2 * theta)) + mu
    return a, b


def plotting_spread(xt, ths, sig):
    break_point = xt[(sig.diff() != 0) & (sig != 0)]
    plt.plot(xt, color='blueviolet')
    plt.axhline(y=ths[0], linestyle='--')
    plt.axhline(y=ths[1], linestyle='--')
    plt.scatter(break_point.index, break_point, color='black', zorder=6, s=100, marker='*')
    plt.show()
    return break_point


def process_signal(raw_signal):
    raw_signal['All'] = raw_signal['Long'].fillna(0) + raw_signal['Short'].fillna(0)
    raw_signal['All'] = raw_signal['All'].replace(0, np.nan).ffill().fillna(0)
    raw_signal = raw_signal.shift(1).fillna(0)
    return raw_signal['All']


def signals(xt, ths):
    Short_P_Long_Q = pd.Series(np.where((xt >= ths[0]), -1, np.nan), index=xt.index, name='Short')
    Long_P_Short_Q = pd.Series(np.where((xt <= ths[1]), 1, np.nan), index=xt.index, name='Long')
    position = pd.concat([Long_P_Short_Q, Short_P_Long_Q], axis=1)
    position = process_signal(position)
    return position


def returns(data: pd.DataFrame, signal: pd.DataFrame, fee):
    ret_df = pd.concat([data['Close'], signal], axis=1).dropna()
    ret_df.columns = ['Close', 'Sig']
    break_point = ret_df['Close'][(ret_df['Sig'].diff() != 0) & (ret_df['Sig'] != 0)]
    print(break_point.index)
    last_trade_return = -fee
    output_df = pd.DataFrame()
    for j in range(len(break_point) - 1):
        start = break_point.index[j]
        end = break_point.index[j+1]
        interval = ret_df[start:end]
        interval = interval.assign(Open=break_point[j])
        interval['Sig'][-1] = interval['Sig'][-1] * -1
        interval['return'] = ((interval['Close'] / interval['Open']) - 1) * interval['Sig'] + last_trade_return
        last_trade_return = interval['return'][-1] - fee * 2
        output_df = pd.concat([output_df, interval])
    output_df = output_df[~output_df.index.duplicated()]
    return output_df['return']


def PnL(data: pd.DataFrame):
    plt.plot(data['P'], linewidth=1.5, color='#d62728', label=P_name)
    plt.plot(data['Q'], linewidth=1.5, color='#1f77b4', label=Q_name)
    plt.plot(data['Total'], linewidth=2, color='black', label='strategy')
    plt.fill_between(DrawDown(data['Total']).index, DrawDown(data['Total']), 0, label='strategy drawdown',
                     color='black', alpha=0.3, hatch='///')
    plt.fill_between(DrawDown(data['P']).index, DrawDown(data['P']), 0, label=P_name + ' drowdown',
                     color='#d62728', alpha=0.15)
    plt.fill_between(DrawDown(data['Q']).index, DrawDown(data['Q']), 0, label=Q_name + ' drawdown',
                     color='#1f77b4', alpha=0.15)
    newHigh = pd.Series(np.where((data['Total'].cummax() - data['Total'] == 0),
                                 data['Total'], np.nan), index=data.index)
    newHigh = newHigh.replace(0, np.nan).dropna()
    plt.scatter(newHigh.index, newHigh, color='lime', zorder=6, s=12, label='new peak')
    plt.legend()
    plt.show()


# 最大風險回撤函數
def DrawDown(data: pd.DataFrame):
    return -(data.dropna().cummax() - data.dropna())


def max_recovery(series):
    series = pd.Series(series)
    series_accu = series.cumsum()
    peak = series_accu[0]
    idx = 0
    rd, mrd = 0, 0
    for i, r in enumerate(series_accu):
        if r > peak:
            peak = r
            rd = i - idx
            idx = i
        if rd > mrd:
            mrd = rd
    return mrd


# 各種比率
class Ratio:
    def __init__(self, series: pd.DataFrame):
        self.data = series

    def Cumulative_Return(self):
        return self.data.dropna()[-1]

    def Annual_Return(self):
        return ((1 + self.data.diff().mean()) ** 252)-1

    def Annual_Volatility(self):
        return self.data.diff().std() * (252 ** 0.5)

    def MDD_Return(self):
        DD = -DrawDown(self.data)
        MDD = max(DD)
        return self.data.dropna()[-1] / MDD

    def Sharpe(self):
        rf = 0
        return ((self.data.diff().mean() - rf)/self.data.diff().std()) * (252 ** 0.5)

    def Sortino(self):
        rf = 0
        return ((self.data.diff().mean() - rf)/self.data.diff()[self.data.diff() < 0].std()) * (252 ** 0.5)


class Performance:
    def __init__(self, pnl):
        self.Total = pnl['Total']
        self.P = pnl['P']
        self.Q = pnl['Q']

    @staticmethod
    def display(series: pd.DataFrame):
        temp_dict = {
            'Total Return': str(round(Ratio(series).Cumulative_Return() * 100, 2)) + ' %',
            'Yearly Return': str(round(Ratio(series).Annual_Return() * 100, 2)) + ' %',
            'Yearly Volatility': str(round(Ratio(series).Annual_Volatility() * 100, 2)) + ' %',
            'MDD': str(round(max(-DrawDown(series)) * 100, 2)) + ' %',
            'ADD': str(round(-DrawDown(series).mean() * 100, 2)) + ' %',
            'Return on MDD': str(round(Ratio(series).MDD_Return(), 2)),
            'Max recovery days': str(max_recovery(series)),
            'Sharpe Ratio': str(round(Ratio(series).Sharpe(), 2)),
            'Sortino Ratio': str(round(Ratio(series).Sortino(), 2))
        }
        return pd.DataFrame(list(temp_dict.items()), columns=['indicator', 'value'])

    def Total_return(self):
        return self.display(self.Total)

    def P_return(self):
        return self.display(self.P)

    def Q_return(self):
        return self.display(self.Q)

    def table(self):
        t = pd.concat([self.Total_return(), self.P_return()['value'], self.Q_return()['value']], axis=1)
        t.set_index('indicator', inplace=True)
        t.columns = ['Stategy', P_name, Q_name]
        print('=' * 60)
        return t


# 論文中的回測方法
def back_test_method_1(ts, te, cost):
    Asset_P = YahooData(P_name, ts[0], te[-1])
    Asset_Q = YahooData(Q_name, ts[0], te[-1])
    Pt, Qt = Asset_P['Close'], Asset_Q['Close']
    beta = co_integration(Pt, Qt)
    spread = Xt(Pt, Qt, beta)
    threshold = threshold_finder(spread, cost)
    sig = signals(spread, threshold)
    plotting_spread(spread, threshold, sig)

    weight_P = 1 / (beta + 1)
    weight_Q = beta / (beta + 1)
    return_P = returns(Asset_P, sig, cost) * weight_P
    return_Q = returns(Asset_Q, -sig, cost) * weight_Q
    return_data = pd.concat([return_P, return_Q, return_P + return_Q], axis=1)
    return_data.columns = ['P', 'Q', 'Total']
    return return_data


# 實際上trade得到的回測方法
def back_test_method_2(fs, fe, ts, te, cost):

    def temp_return(Pt, Qt, spread, thresholds):
        sig = signals(spread, thresholds)
        plotting_spread(spread, thresholds, sig)

        weight_P = 1 / (Fitting_beta + 1)
        weight_Q = Fitting_beta / (Fitting_beta + 1)
        return_P = returns(Pt, sig, cost) * weight_P
        return_Q = returns(Qt, -sig, cost) * weight_Q
        return_data = pd.concat([return_P, return_Q, return_P + return_Q], axis=1)
        return_data.columns = ['P', 'Q', 'Total']
        return return_data

    result_df = pd.DataFrame()
    last_P = 0
    last_Q = 0
    last_total = 0

    for ind in range(len(fs)):
        # Fitting_data
        Fitting_Asset_P = YahooData(P_name, fs[ind], fe[ind])
        Fitting_Asset_Q = YahooData(Q_name, fs[ind], fe[ind])
        F_Pt, F_Qt = Fitting_Asset_P['Close'], Fitting_Asset_Q['Close']

        # Testing_data
        Testing_Asset_P = YahooData(P_name, ts[ind], te[ind])
        Testing_Asset_Q = YahooData(Q_name, ts[ind], te[ind])
        T_Pt, T_Qt = Testing_Asset_P['Close'], Testing_Asset_Q['Close']

        # Testing data 開始前一天(Fitting data 的最後一天) 所迴歸出的Fitting_beta
        Fitting_beta = co_integration(F_Pt, F_Qt)
        print(Fitting_beta)

        # Testing data 開始前一天(Fitting data 的最後一天) 所算出的thresholds
        Fitting_spread = Xt(F_Pt, F_Qt, Fitting_beta)
        Fitting_thresholds = threshold_finder(Fitting_spread, cost)
        print('a=', Fitting_thresholds[0], ', b=', Fitting_thresholds[1])

        # 將Fitting beta 帶入算出 Test data價差
        Testing_spread = Xt(T_Pt, T_Qt, Fitting_beta)

        temp_df = temp_return(Testing_Asset_P, Testing_Asset_Q, Testing_spread, Fitting_thresholds)
        temp_df['P'] = temp_df['P'] + last_P
        temp_df['Q'] = temp_df['Q'] + last_Q
        temp_df['Total'] = temp_df['Total'] + last_total
        result_df = pd.concat([result_df, temp_df])
        last_P = temp_df['P'][-1]
        last_Q = temp_df['Q'][-1]
        last_total = temp_df['Total'][-1]
    return result_df


if __name__ == '__main__':
    P_name = '0056.TW'
    Q_name = '0050.TW'
    # 設定手續費
    cost_fee = 0.00471

    FS_Year = ['2011', '2012', '2013', '2014', '2015']
    FE_Year = ['2015', '2016', '2017', '2018', '2019']
    TS_Year = ['2015', '2016', '2017', '2018', '2019']
    TE_Year = ['2016', '2017', '2018', '2019', '2020']
    month = '-09'
    date = '-23'
    date_plus_one = '-24'
    Fitting_start = []
    Fitting_end = []
    Testing_start = []
    Testing_end = []
    for i in range(len(FS_Year)):
        Fitting_start.append(FS_Year[i] + month + date)
        Fitting_end.append(FE_Year[i] + month + date)
        Testing_start.append(TS_Year[i] + month + date_plus_one)
        Testing_end.append(TE_Year[i] + month + date)

    return_df1 = back_test_method_1(Testing_start, Testing_end, cost_fee)
    try:
        return_df2 = back_test_method_2(Fitting_start, Fitting_end, Testing_start, Testing_end, cost_fee)
        return_df = pd.concat([return_df1, return_df2], axis=1).ffill()
        return_df.columns = ['paper P', 'paper Q', 'paper total', 'P', 'Q', 'Total']
        PnL(return_df)
        plt.plot(return_df['paper total'], label='method 1',  color='#d62728')
        plt.plot(return_df['Total'], label='method 2', color='#1f77b4')
        plt.fill_between(DrawDown(return_df['paper total']).index, DrawDown(return_df['paper total']), 0, label='method 1 drawdown',
                         color='#d62728', alpha=0.3, hatch='///')
        plt.fill_between(DrawDown(return_df['Total']).index, DrawDown(return_df['Total']), 0, label='method 2 drawdown',
                         color='#1f77b4', alpha=0.3, hatch='///')
        newHigh = pd.Series(np.where((return_df['Total'].cummax() - return_df['Total'] == 0),
                                     return_df['Total'], np.nan), index=return_df.index)
        newHigh = newHigh.replace(0, np.nan).dropna()
        plt.scatter(newHigh.index, newHigh, color='lime', zorder=6, s=12, label='new peak')
        plt.legend()
        plt.show()
        pfc = Performance(return_df).table()
        print(pfc)
    except KeyError:
        print('交易不到啦幹')



