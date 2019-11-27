import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import pickle
import re


def black(fwd , k, vol , r, t, opt_type='call'):
    d1 = (np.log(fwd / k) + (vol**2/2)*t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    if opt_type == 'call':
        opt_px = np.exp(-r * t) * (fwd * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        opt_px = np.exp(-r * t) * (k * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    return opt_px


def delta(fwd , k, vol , r, t, opt_type='call'):
    d1 = (np.log(fwd / k) + (vol**2/2)*t) / (vol * np.sqrt(t))
    if opt_type == 'call':
        delta = np.exp(-r * t) * norm.cdf(d1)
    else:
        delta = np.exp(-r * t) * (norm.cdf(d1) - 1)
    return delta


def gamma(fwd , k, vol , r, t):
    return delta(fwd , k, vol , r, t)/(fwd*vol*np.sqrt(t))


def imp_vol(opt_px, fwd , k , r, t, opt_type='call'):
    if opt_type == 'call':
        vol = newton( lambda vol : black(fwd , k, vol , r, t ) - opt_px , 0.2)
    else:
        vol = newton( lambda vol : black(fwd , k, vol , r, t ,'put') - opt_px , 0.2)
    return vol


class CMECalendar:

    IMM_dates = pd.to_datetime(['18-DEC-15','18-MAR-16','17-JUN-16','16-SEP-16','16-DEC-16',
                '17-MAR-17','16-JUN-17','15-SEP-17','15-DEC-17','16-MAR-18','15-JUN-18',
               '21-SEP-18','21-DEC-18','15-MAR-19','21-JUN-19','20-SEP-19','20-DEC-19', '20-MAR-20'])
    start = pd.to_datetime('31-DEC-15')
    end = date(start.year + 5, 3, 31)
    EOM_dates = pd.date_range(start, end, freq='BM')


class RatesCurve:

    def __init__(self, path='data/libor_1m.csv'):
        self.rates = pd.read_csv(path)
        self.rates.DATE = pd.to_datetime(self.rates.DATE)
        self.rates = self.rates.set_index('DATE')
        self.rates['LIBOR'] = self.rates['LIBOR'].astype(float)

    def get(self, date):
        current_date = pd.to_datetime(date)
        return float(self.rates.loc[current_date])


class FuturesCurve:

    def __init__(self, path='data/fut.pkl'):
        self.instrument_list = ['ES', 'NQ', 'CD', 'EC', 'JY', 'MP', 'TY', 'US', 'C', 'S', 'W', 'CL', 'GC']
        self.df = self.load(path)
        self.col_dict = {inst: [key for key in self.df.columns if re.match(r"{}_+".format(inst), key)]
                         for inst in self.instrument_list}

    def get(self, inst, today, fut_date):
        if fut_date == '1W':
            return self.df[self.col_dict[inst][0]][today]
        if fut_date == '1M':
            return self.df[self.col_dict[inst][1]][today]
        if fut_date == '2M':
            return self.df[self.col_dict[inst][2]][today]
        if pd.to_datetime(fut_date) <= pd.to_datetime(today) + timedelta(weeks=1):
            return self.df[self.col_dict[inst][0]][today]
        elif pd.to_datetime(fut_date) <= pd.to_datetime(today) + relativedelta(months=1):
            t0 = 0
            t1 =((pd.to_datetime(today) + relativedelta(months=1)) - pd.to_datetime(today)).days
            t = (pd.to_datetime(fut_date) - pd.to_datetime(today)).days
            f = ((t-t0)/t1) * self.df[self.col_dict[inst][1]][today] + \
                ((t1-t)/t1) * self.df[self.col_dict[inst][0]][today]
            return f
        elif pd.to_datetime(fut_date) <= pd.to_datetime(today) + relativedelta(months=2):
            t0 = 0
            t1 = ((pd.to_datetime(today) + relativedelta(months=2)) -
                  (pd.to_datetime(today) + relativedelta(months=1))).days
            t = (pd.to_datetime(fut_date)  - (pd.to_datetime(today) + relativedelta(months=1))).days
            f = ((t-t0)/t1) * self.df[self.col_dict[inst][2]][today] + \
                ((t1-t)/t1) * self.df[self.col_dict[inst][1]][today]
            return f
        else:
            raise IndexError('Date must be within two months of today')

    def load(self, path):
        pickle_in = open(path, "rb")
        fut_pd = pickle.load(pickle_in)
        pickle_in.close()
        return fut_pd.fillna(fut_pd.mean())


class VolCurve:
    def __init__(self):
        self.vol_poly = self.load('data/vol_poly.pkl')
        self.rate_curve = RatesCurve()
        self.futures_curve = FuturesCurve()

    def load(self, path):
        pickle_in = open(path, "rb")
        vol_poly = pickle.load(pickle_in)
        pickle_in.close()
        return vol_poly

class CreateVolCurve:

    def __init__(self, instrument, today, IMM1_call_prices, IMM1_put_prices,
                              IMM2_call_prices, IMM2_put_prices,
                              ):

        self.today = pd.to_datetime(today)
        self.instrument = instrument
        self.rate_curve = RatesCurve()
        self.calendar = CMECalendar()
        self.opt_prices = {}
        self.fut_prices = {}
        exp_date1 = self.today
        for i, d in enumerate(self.calendar.IMM_dates):
            if exp_date1 < d:
                exp_date1 = d
                break
        exp_date2 = self.calendar.IMM_dates[i + 1]
        self.expiration = {'IMM1_call': exp_date1, 'IMM1_put': exp_date1,
                           'IMM2_call': exp_date2, 'IMM2_put': exp_date2}
        self.parse(IMM1_call_prices, 'IMM1_call')
        self.parse(IMM2_call_prices, 'IMM2_call')
        self.parse(IMM1_put_prices, 'IMM1_put')
        self.parse(IMM2_put_prices, 'IMM2_put')
        self.vol_curve = self.calc_ivols()

    def parse(self, file, exp):
        df = pd.read_csv(file)
        self.opt_prices[exp] = df[['strike', 'settle']]
        self.fut_prices[exp] = df.future[0]

    def calc_ivols(self):
        vc = {}
        for i, exp in enumerate(['IMM1', 'IMM2']):
            vol_dict = {}
            # Calculate option moneyness
            ind_call = (self.fut_prices[exp + '_call'] * 1 < self.opt_prices[exp + '_call'].strike
                        ) &  (self.opt_prices[exp + '_call'].strike < self.fut_prices[exp + '_call'] * 1.2)
            ind_put = (self.fut_prices[exp + '_put'] * 0.8 < self.opt_prices[exp + '_put'].strike
                       ) & (self.opt_prices[exp + '_put'].strike < self.fut_prices[exp + '_put'] * 1)

            calls = self.opt_prices[exp + '_call'][ind_call]
            puts = self.opt_prices[exp + '_put'][ind_put]

            for j in range(puts.shape[0]):
                xp = exp + '_put'
                t = self.expiration[xp] - self.today
                t = t.days
                vol_dict[puts.strike.iloc[j]] = imp_vol(puts.settle.iloc[j],
                                                        self.fut_prices[xp],
                                                        puts.strike.iloc[j],
                                                        self.rate_curve.get(self.today) * 0.01,
                                                        t / 365,
                                                        opt_type='put')
            for j in range(calls.shape[0]):
                xp = exp + '_call'
                t = self.expiration[xp] - self.today
                t = t.days
                vol_dict[calls.strike.iloc[j]] = imp_vol(calls.settle.iloc[j],
                                                        self.fut_prices[xp],
                                                        calls.strike.iloc[j],
                                                        self.rate_curve.get(self.today) * 0.01,
                                                        t / 365,
                                                        opt_type='call')
            vc[exp] = pd.DataFrame(vol_dict.values(), index= vol_dict.keys(), columns = ['imp_vol'])
        return vc