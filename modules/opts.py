import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
from numpy.polynomial.polynomial import polyval
import pickle
import re


def black(fwd , k, vol , r, t, opt_type='call'):
    """
    This function calculates option prices through Black-76 model
    Source: https://en.wikipedia.org/wiki/Black_model
    :param fwd: forward price of underlying
    :param k: strike
    :param vol: volatility
    :param r: interest rate
    :param t: time to expiration
    :param opt_type: option type (call or put)
    :return: option price
    """
    d1 = (np.log(fwd / k) + (vol**2/2)*t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    if opt_type == 'call':
        opt_px = np.exp(-r * t) * (fwd * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        opt_px = np.exp(-r * t) * (k * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    return opt_px


def delta(fwd , k, vol , r, t, opt_type='call'):
    """
    This function calculates the delta of an option assuming Black-76 model
    Source: https://www.glynholton.com/notes/black_1976/
    :param fwd: forward price of underlying
    :param k: strike
    :param vol: volatility
    :param r: interest rate
    :param t: time to expiration
    :param opt_type: option type (call or put)
    :return: delta
    """
    d1 = (np.log(fwd / k) + (vol**2/2)*t) / (vol * np.sqrt(t))
    if opt_type == 'call':
        delta = np.exp(-r * t) * norm.cdf(d1)
    else:
        delta = np.exp(-r * t) * (norm.cdf(d1) - 1)
    return delta


def gamma(fwd , k, vol , r, t):
    """
    This function calculates the gamma of an option assuming Black-76 model
    Source: https://www.glynholton.com/notes/black_1976/
    :param fwd: forward price of underlying
    :param k: strike
    :param vol: volatility
    :param r: interest rate
    :param t: time to expiration
    :return: gamma
    """
    d1 = (np.log(fwd / k) + (vol ** 2 / 2) * t) / (vol * np.sqrt(t))
    return np.exp(-r * t) * norm.pdf(d1)


def imp_vol(opt_px, fwd, k, r, t, opt_type='call'):
    """
    This function calculates the implied volatility of an option assuming Black-76 model
    :param opt_px: option price
    :param fwd: forward price of underlying
    :param k: strike
    :param r: interest rate
    :param t: time to expiration
    :param opt_type: option type (call or put)
    :return: implied volatility
    """
    if opt_type == 'call':
        vol = newton( lambda vol : black(fwd , k, vol , r, t ) - opt_px , 0.2)
    else:
        vol = newton( lambda vol : black(fwd , k, vol , r, t ,'put') - opt_px , 0.2)
    return vol


def delta_moneyness(x, vol, r, t, opt_type='call'):
    x = np.exp(x)
    if opt_type == 'call':
        return delta(1, x, vol, r, t, opt_type=opt_type)
    else:
        return delta(1, x, vol, r, t, opt_type=opt_type)


def find_delta(vol_poly, r, t, delta=0.25, opt_type='call'):
    x = newton(lambda x : abs(delta_moneyness(x, polyval(x,vol_poly), r, t, opt_type=opt_type)) - delta, 0)
    return float(x), float(polyval(x,vol_poly))


class CMECalendar:
    """
    IMM_dates: Options expirations for options that expire on IMM dates, 3rd Friday of each month
    minus a day if holiday
    EOM_dates: Options expirations for options that expire on the last business day of every month
    """

    IMM_dates = pd.to_datetime(['18-DEC-15','18-MAR-16','17-JUN-16','16-SEP-16','16-DEC-16',
                '17-MAR-17','16-JUN-17','15-SEP-17','15-DEC-17','16-MAR-18','15-JUN-18',
               '21-SEP-18','21-DEC-18','15-MAR-19','21-JUN-19','20-SEP-19','20-DEC-19', '20-MAR-20'])
    start = pd.to_datetime('31-DEC-15')
    end = date(start.year + 5, 3, 31)
    EOM_dates = pd.date_range(start, end, freq='BM')


class RatesCurve:
    """
    This class obtains 1 month libor's close.
    Work in progress: Add 3 month libor and strip a curve for better results
    In Black model, if you have the futures price, interest rates only play a factor
    through discounting, so having a less-than-perfect estimate of the relevant interest
    rate isn't as important as in, say, Black-Scholes.
    """
    def __init__(self):
        self.rates = pd.read_csv('data/libor_1m.csv')
        self.rates.DATE = pd.to_datetime(self.rates.DATE)
        self.rates = self.rates.set_index('DATE')
        self.rates['LIBOR'] = self.rates['LIBOR'].astype(float)

        self.rates_1m = self.rates

        self.rates_3m = pd.read_csv('data/libor_3m.csv')
        self.rates_3m.DATE = pd.to_datetime(self.rates_3m.DATE)
        self.rates_3m = self.rates_3m.set_index('DATE')
        self.rates_3m['LIBOR'] = self.rates_3m['LIBOR'].astype(float)

        self.rates_6m = pd.read_csv('data/libor_6m.csv')
        self.rates_6m.DATE = pd.to_datetime(self.rates_6m.DATE)
        self.rates_6m = self.rates_6m.set_index('DATE')
        self.rates_6m['LIBOR'] = self.rates_6m['LIBOR'].astype(float)

    def get(self, today, fut_date='' ):
        """
        Get 1 month libor for a specific date
        :param today: date in any format that can be converted to datetime
        :return: 1 month libor (float)
        """
        current_date = pd.to_datetime(today)
        if fut_date== '':
            return float(self.rates.loc[current_date])
        else:
            fut_date = pd.to_datetime(fut_date)
            if fut_date <= current_date + relativedelta(months=1):
                return float(self.rates.loc[current_date])
            elif fut_date <= current_date + relativedelta(months=3):
                t0 = 0
                t1 = 91
                t = (pd.to_datetime(fut_date) - pd.to_datetime(current_date)).days
                r = ((t - t0) / t1) * float(self.rates_3m.loc[current_date]) + \
                    ((t1 - t) / t1) * float(self.rates.loc[current_date])
                return r
            elif fut_date <= current_date + relativedelta(months=6):
                t0 = 0
                t1 = 182
                t = (pd.to_datetime(fut_date) - pd.to_datetime(current_date)).days
                r = ((t - t0) / t1) * float(self.rates_6m.loc[current_date]) + \
                    ((t1 - t) / t1) * float(self.rates_3m.loc[current_date])
                return r
            else:
                return float(self.rates_6m.loc[current_date])


class FuturesCurve:
    """
    FuturesCurve reads a database of 1 week, 1 month, and 2 month futures prices and interpolates to obtain
    futures prices for any date inside 2 months.
    """

    def __init__(self, path='data/fut.pkl'):
        self.instrument_list = ['ES', 'NQ', 'CD', 'EC', 'JY', 'MP', 'TY', 'US', 'C', 'S', 'W', 'CL', 'GC']
        self.df = self.load(path)
        self.col_dict = {inst: [key for key in self.df.columns if re.match(r"{}_+".format(inst), key)]
                         for inst in self.instrument_list}

    def get(self, inst, today, fut_date):
        """
        This function returns the futures price for a specific asset, date, and expiry, as long as expiry
        is within two months of date.
        :param inst: Instrument: ['ES', 'NQ', 'CD', 'EC', 'JY', 'MP', 'TY', 'US', 'C', 'S', 'W', 'CL', 'GC']
        :param today: date in any format that can be converted to datetime
        :param fut_date: Expiry date. if '1W', '1M', or '2M', obtain price for 1 week, 1 month or 2 months,
        respectively. Else, expiry date in any format that can be converted to datetime.
        :return: futures price
        """
        # Get price directly from database
        if fut_date == '1W':
            return self.df[self.col_dict[inst][0]][today]
        if fut_date == '1M':
            return self.df[self.col_dict[inst][1]][today]
        if fut_date == '2M':
            return self.df[self.col_dict[inst][2]][today]

        # If under 1 week, assume price is 1 week price. We won't use prices under 1 week in any
        # calculation with this function
        if pd.to_datetime(fut_date) <= pd.to_datetime(today) + timedelta(weeks=1):
            return self.df[self.col_dict[inst][0]][today]

        # If within 1 month, interpolate between 1 week and 1 month
        elif pd.to_datetime(fut_date) <= pd.to_datetime(today) + relativedelta(months=1):
            t0 = 0
            t1 =((pd.to_datetime(today) + relativedelta(months=1)) - pd.to_datetime(today)).days
            t = (pd.to_datetime(fut_date) - pd.to_datetime(today)).days
            f = ((t-t0)/t1) * self.df[self.col_dict[inst][1]][today] + \
                ((t1-t)/t1) * self.df[self.col_dict[inst][0]][today]
            return f

        # If under two months, interpolate between 1 month and 2 month
        elif pd.to_datetime(fut_date) <= pd.to_datetime(today) + relativedelta(months=2) +timedelta(days=2):
            t0 = 0
            t1 = ((pd.to_datetime(today) + relativedelta(months=2)) +timedelta(days=2) -
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
    """
    This class has vol_poly, a dictionary with instrument codes as keys and values a pandas
    dataframe with indices as dates, and columns as the oefficients of a 5 degree polynomial
    which represents the volatility surface.
    """
    def __init__(self):
        self.vol_poly_1M = self.load('data/vol_poly_1M.pkl')
        self.vol_poly_2M = self.load('data/vol_poly_2M.pkl')
        self.rate_curve = RatesCurve()
        self.futures_curve = FuturesCurve()

    def load(self, path):
        pickle_in = open(path, "rb")
        vol_poly = pickle.load(pickle_in)
        pickle_in.close()
        return vol_poly


class VolCurveAgg:

    def __init__(self, instrument, today, vol_dict):

        self.instrument = instrument
        self.today = pd.to_datetime(today)
        self.vol_dict = vol_dict
        self.rate_curve = RatesCurve()
        self.fut_prices = {}
        self.vol_curve = self.calc_ivols()
        self.up_gamma = 0
        self.down_gamma = 0
        self.up_gamma_5 = 0
        self.down_gamma_5 = 0
        self.agg_gamma()
        self.features = self.calc_features()

    def calc_features(self):
        features = pd.DataFrame(columns=[self.instrument + '_up_gamma',
                                              self.instrument + '_up_gamma_5',
                                              self.instrument + '_down_gamma',
                                              self.instrument + '_down_gamma_5'],
                                index=[self.today])
        features.loc[self.today] = [self.up_gamma, self.up_gamma_5, self.down_gamma,self.down_gamma_5]
        return features

    def agg_gamma(self):
        for key in self.vol_curve.keys():
            r = self.rate_curve.get(self.today, key)
            t = (key - self.today).days / 365
            for strike in self.vol_curve[key].index:
                if strike >= 1.025 * self.fut_prices[key]:
                    self.up_gamma += self.vol_curve[key].oi.loc[strike] * gamma(self.fut_prices[key],
                                      strike,
                                      self.vol_curve[key].imp_vol.loc[strike],
                                      r,
                                      t)
                    self.up_gamma_5 += self.vol_curve[key].oi.loc[strike] * gamma(1.05 * self.fut_prices[key],
                                      strike,
                                      self.vol_curve[key].imp_vol.loc[strike],
                                      r,
                                      t)
                if strike <= 0.975 * self.fut_prices[key]:
                    self.down_gamma_5 += self.vol_curve[key].oi.loc[strike] * gamma(0.95 * self.fut_prices[key],
                                      strike,
                                      self.vol_curve[key].imp_vol.loc[strike],
                                      r,
                                      t)
                    self.down_gamma += self.vol_curve[key].oi.loc[strike] * gamma(self.fut_prices[key],
                                      strike,
                                      self.vol_curve[key].imp_vol.loc[strike],
                                      r,
                                      t)

    def calc_ivols(self):
        vc = {}
        for key in self.vol_dict[self.today]['Call'].keys():

            # See if no options data for a particular expiration date, if missing for either calls or puts,
            # ignore date
            try:
                if (str(self.vol_dict[self.today]['Call'][key]) == '') | (str(self.vol_dict[self.today]['Put'][key]) == ''):
                    pass
                else:
                    imp_vol_dict = {}

                # Calculate options moneyness. Only consider 80%-120% moneyness (fut/strike)
                    fut = self.vol_dict[self.today]['Call'][key].future.iloc[0]
                    expiration = self.vol_dict[self.today]['Call'][key].expiration.iloc[0]
                    # saving futures price ina  dictionary for gamma calculation
                    self.fut_prices[expiration] = fut
                    # Above 1 moneyness consider calls, under consider puts
                    ind_call = (fut * 1 < self.vol_dict[self.today]['Call'][key].strike
                                ) &  (self.vol_dict[self.today]['Call'][key].strike < fut * 1.2)
                    ind_put = (fut * 0.8 < self.vol_dict[self.today]['Put'][key].strike
                                ) &  (self.vol_dict[self.today]['Put'][key].strike < fut * 1)

                    calls = self.vol_dict[self.today]['Call'][key].loc[ind_call]
                    puts = self.vol_dict[self.today]['Put'][key].loc[ind_put]

                    # cycle through all puts and calculate implied volatility
                    for j in range(puts.shape[0]):
                        t = expiration - self.today
                        t = t.days
                        imp_vol_dict[puts.strike.iloc[j]] = [imp_vol(puts.settle.iloc[j],
                                                                fut,
                                                                puts.strike.iloc[j],
                                                                self.rate_curve.get(self.today) * 0.01,
                                                                t / 365,
                                                                opt_type='put'), puts.oi.iloc[j]]

                    # cycle through all calls and calculate implied volatility
                    for j in range(calls.shape[0]):
                        t = expiration - self.today
                        t = t.days
                        imp_vol_dict[calls.strike.iloc[j]] = [imp_vol(calls.settle.iloc[j],
                                                                fut,
                                                                calls.strike.iloc[j],
                                                                self.rate_curve.get(self.today) * 0.01,
                                                                t / 365,
                                                                opt_type='call'), calls.oi.iloc[j]]
                    vc[expiration] = pd.DataFrame(imp_vol_dict.values(),
                                                  index= imp_vol_dict.keys(),
                                                  columns = ['imp_vol','oi'])
            except:
                pass
        return vc


class CreateVolCurveSample:

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