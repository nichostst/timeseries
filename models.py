import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings


class UnivariateTSModel:
    def __init__(self, data):
        self.data = data
        self.rmse = None
        self.fitted = False
        self.predict = None
        self.fcst = None

    def fit(self):
        pass
    
    def predict(self, extend):
        pass

    def plot(self, predict=True):
        '''
        Plot fitted data

        Inputs
        ======
        predict: bool, whether to plot prediction (fit) or forecast
        '''
        plt.figure(figsize=(10,5))
        plt.plot(self.data.index, self.data.values)
        if predict:
            plt.plot(self.predict.index, self.predict.values)
        elif self.fcst is None:
            raise Exception("Run forecast method before using predict=False")
        else:
            plt.plot(self.fcst.index, self.fcst.values)

        plt.title('Monthly vs Fitted Sales')
        plt.legend(['Sales', 'Forecast'])
    
    def acf_plot(self):
        '''
        Get plot of ACF given data
        '''
        n = len(self.data)
        lag = np.array([i for i in range(1, self.period)])
        acf = np.array([np.corrcoef(self.data[:-i], self.data[i:])[0,1]
                        for i in lag])
        # Cutoffs
        cutoff = 2/np.sqrt(n-lag)

        plt.figure(figsize=(7,4))
        plt.ylim(-1,1)
        plt.bar(lag, acf, width=0.3, alpha=0.6)
        plt.fill_between(lag, cutoff, -cutoff, alpha=0.4, color='skyblue')
        plt.title('ACF Plot')


class BrownExpSmoothing(UnivariateTSModel):
    '''
    Seasonal-adjusted Brown Exponential Smoothing
    '''
    def __init__(self, data, period=12):
        super(BrownExpSmoothing, self).__init__(data)
        self.adjusted_data = None
        self.alpha = None
        self.seasonal = True
        self.period = period

    @staticmethod
    def get_les_forecast(data, alpha, extend=0):
        '''
        Get forecasts for linear exponential smoothing (LES) model.

        Inputs
        ======
        data: Pandas Series object
        alpha: float (0 to 1), smoothing parameter
        extend: positive integer, to forecast

        Outputs
        =======
        f: Forecast
        e: Errors
        '''
        # Initialize extended data (only applicable if extended),
        # errors, and forecasts
        idx = [x for x in range(len(data)+extend)]
        d = pd.Series(0, index=idx, dtype=np.float32)
        d[[x for x in range(len(data))]] = data
        e = pd.Series(0, index=idx, dtype=np.float32)
        f = pd.Series(0, index=idx, dtype=np.float32)

        # Forecast base cases
        f[0] = d.iloc[0]
        f[1] = d.iloc[0]

        # Base errors
        e[0] = d.iloc[0] - f[0]
        e[1] = d.iloc[1] - f[1]

        for i in range(2, len(data)+extend):
            L = 2*d.iloc[i-1] - d.iloc[i-2]
            S = -2*(1-alpha)*e[i-1] + (1-alpha)**2*e[i-2]
            # Forecast and error
            f[i] = L+S
            # Extended data
            if (extend != 0) and (i >= len(data)):
                d[i] = f[i]
                e[i] = 0
            else:
                e[i] = d.iloc[i] - f[i]

        return f, e, d


    @staticmethod
    def get_seasonal_index(data, period):
        '''
        Get seasonal adjustment index for linear exponential
        smoothing (LES) model.

        Inputs
        ======
        data: Pandas Series object

        Outputs
        =======
        seasonal_index: NumPy array, seasonal adjustment index
        '''
        p = period/2
        trend_cycle = (
            data.rolling(period).mean().shift(-int(np.ceil(p)))
            + data.rolling(period).mean().shift(-int(np.floor(p)))
        )/2

        # Ratio of unit sold to centered moving average
        ratio = data.values/trend_cycle

        # Get unnormalized seasonal index
        foo = [ratio[period*i:period*(i+1)].values
               for i in range(int(len(ratio)/period)+1)]
        un_index = np.nanmean([np.append(x, np.repeat(np.nan, period-len(x)))
                               for x in foo], axis=0)

        # Normalizing to get seasonal index
        norm_index = un_index/un_index.sum()*period
        seasonal_index = np.tile(norm_index,
                                 int(len(ratio)/period)+1)[:len(ratio)]
        return seasonal_index


    @staticmethod
    def get_optimized_alpha(data, period, init=0.5):
        '''
        Least-squares linear regression for equally-spaced data.

        Inputs
        ======
        data: Pandas Series object
        init: float (0 to 1), initial guess for alpha

        Outputs
        =======
        alpha: float, optimized alpha
        rmse: float, optimized RMSE
        '''
        seasonal_index = BrownExpSmoothing.get_seasonal_index(data, period)
        # Adjust data by seasonal index
        adjusted_data = data.values/seasonal_index
        # Define function to minimize
        def func(alpha):
            f, e, _ = BrownExpSmoothing.get_les_forecast(adjusted_data, alpha)

            # Reseasonalize
            forecast = f*seasonal_index
            forecast_error = data.values - forecast
            rmse = np.sqrt(np.mean(forecast_error**2))
            return rmse

        # Nelder-Mead downhill simplex method
        op = minimize(lambda x: func(x), init, method='Nelder-Mead')
        alpha = op.x[0]
        rmse = op.fun
        return alpha, rmse
    
    
    def fit(self):
        seasonal_index = self.__class__.get_seasonal_index(self.data,
                                                           self.period)
        # Adjust data by seasonal index
        self.adjusted_data = self.data.values/seasonal_index

        # Get optimized alpha and fit
        self.alpha, self.rmse = self.__class__.get_optimized_alpha(self.data,
                                                                   self.period)
        f, e, _ = self.__class__.get_les_forecast(self.adjusted_data,
                                                  self.alpha)
        les_forecast = f
        les_error = e

        # Reseasonalize
        self.predict = pd.Series(les_forecast.values*seasonal_index,
                                 index=self.data.index)
        self.fitted = True


    def forecast(self, extend=12):
        if self.fitted:
            seasonal_index = self.__class__.get_seasonal_index(self.data,
                                                               self.period)
            # Forecast the next 15 months
            f, _, _ = self.__class__.get_les_forecast(self.adjusted_data,
                                                      self.alpha,
                                                      extend=extend)

            # Forecast dates
            dates = pd.date_range(self.data.index.min(),
                                  periods=len(f), freq='M')

            # Calculate reseasonalized forecast
            self.fcst = pd.Series([seasonal_index[x.month-1]*y
                                   for x, y in zip(dates, f)], index=dates)
        else:
            raise Exception("Run fit method first!")


class STLDecomp(UnivariateTSModel):
    '''
    Seasonal-Trend decomposition with LOESS Regression
    '''
    def __init__(self, data, period=12):
        super(STLDecomp, self).__init__(data)
        self.seasonal = True
        self.period = period
        self.stl = None
        self.r2 = None


    @staticmethod
    def ls_regression(data, line=True):
        '''
        Least-squares linear regression for equally-spaced data.

        Inputs
        ======
        data: Pandas Series object

        Outputs
        =======
        If line,
            l: Pandas Series object, line of best fit
        Else,
            m: float, slope of the least-squares estimate
            b: float, y-intercept of the least-squares estimate
        '''
        n = len(data)
        x = np.array([i for i in range(n)])
        m = (n*np.sum(x*data) - np.sum(x)*np.sum(data))
        m = m/(n*np.sum(x**2) - np.sum(x)**2)
        b = (np.sum(data) - m*np.sum(x))/n
        if line:
            return pd.Series(m*x+b, name='LSReg')
        return m, b


    @staticmethod
    def extend_forecast(trend, extend=0):
        '''
        Extend trend with linear assumption

        Inputs
        ======
        data: Pandas Series object
        extend: float, extend by

        Outputs
        =======
        Pandas Series object, extended forecast
        '''
        n = len(trend)
        m, b = STLDecomp.ls_regression(trend, line=False)
        x = np.array([i for i in range(n+extend)])
        return m*x+b


    def check_r2(self):
        '''
        Check whether R2 is below 0.95 and warn the user

        Inputs
        ======
        stl: statsmodels.tsa.seasonal.DecomposeResult object
        '''
        y = self.stl.trend
        yhat = self.__class__.ls_regression(self.stl.trend).values
        self.r2 = 1 - np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)

        if self.r2 < 0.95:
            warnings.warn("R2 is below 0.95, prediction accuracy may be bad.")


    def fit(self, extrapolate=1):
        stl = seasonal_decompose(self.data.values, period=self.period,
                                 extrapolate_trend=extrapolate)
        rmse = np.sqrt(np.mean(stl.resid**2))
        self.stl = stl
        self.fitted = True
        self.predict = pd.Series(stl.trend+stl.seasonal, index=self.data.index)
        self.rmse = np.sqrt(np.mean(stl.resid**2))


    def forecast(self, extend=12):
        if self.fitted:
            self.__class__.check_r2(self)
            extended = self.__class__.extend_forecast(self.stl.trend,
                                                      extend=extend)
            dates = pd.date_range(self.data.index.min(),
                                  periods=len(extended), freq='M')

            # Add seasonality to get forecast
            seasonality = np.tile(
                self.stl.seasonal[:12], int(np.ceil(extended.shape[0]/12))
            )[:extended.shape[0]]
            self.fcst = pd.Series(extended+seasonality, index=dates)
        else:
            raise Exception("Run fit method first!")


class HoltWinters(UnivariateTSModel):
    '''
    Holt-Winters exponential smoothing without seasonal smoothing
    '''
    def __init__(self, data, period=12):
        super(HoltWinters, self).__init__(data)
        self.seasonal = True
        self.period = period
        self.model = None


    def fit(self, extrapolate=1):
        self.model = ExponentialSmoothing(
            self.data.values,
            seasonal='add',
            seasonal_periods=self.period
        ).fit(
            smoothing_seasonal=0,
            use_boxcox=True,
            use_basinhopping=True
        )
        self.fitted = True
        self.predict = pd.Series(self.model.predict(0, len(self.data)-1),
                                 index=self.data.index)
        self.rmse = np.sqrt(np.mean((self.data-self.predict)**2))


    def forecast(self, extend=12):
        if self.fitted:
            dates = pd.date_range(self.data.index.min(),
                                  periods=len(self.data)+extend, freq='M')
            self.fcst = pd.Series(self.model.predict(0, len(self.data)+extend-1),
                                  index=dates)
        else:
            raise Exception("Run fit method first!")