import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dtime
from scipy.stats import norm
import streamlit as st
import seaborn as sns

class DataFetch:
    def __init__(self, stock, days, droplist=None):
        self.stock = stock
        self.days = days
        self.droplist = droplist

    def data_pull(self):
        today = dtime.datetime.today()
        start = today - dtime.timedelta(self.days)
        ticker = yf.Ticker(self.stock)
        df = ticker.history(start= start, end= today)
        if df.empty:
            raise ValueError("Please Input A Valid Stock Ticker")
        if self.droplist is not None:
            for col in self.droplist:
                if col in df.columns:
                    df = df.drop(col, axis=1)
        return df

    @staticmethod
    def risk_free():
        ticker = yf.Ticker('^TNX')
        tnx_data = ticker.history(period='1d')
        rf = round(tnx_data['Close'].iloc[-1] / 100, 4)
        return rf


class JumpDiffusion:
    def __init__(self, prices, rf):
        self.prices = prices
        self.rf = rf

    @staticmethod
    def historical_vol(prices):
        ln_returns = np.log(prices / prices.shift(1)).dropna()
        daily_vol = ln_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol

    @staticmethod
    def drift(prices):
        ln_returns = np.log(prices / prices.shift(1)).dropna()
        mean_log_returns = ln_returns.mean()
        var_log_returns = ln_returns.var()
        drift = mean_log_returns - (0.5 * var_log_returns)
        return drift

    @staticmethod
    def jump_parameters(prices):
        prices['returns'] = prices['Close'].pct_change()
        prices['rolling mean'] = prices['returns'].rolling(window=30).mean()
        prices['rolling sd'] = prices['returns'].rolling(window=30).std()
        prices = prices.dropna()

        changes = prices['returns'].abs() > prices['rolling mean'] + 2 * prices['rolling sd']
        jumps = prices.loc[changes]
        lam = len(jumps) / len(prices)

        log_jump_sizes = np.log(1 + jumps['returns'])
        mu_j, sigma_j = norm.fit(log_jump_sizes)
        return lam, mu_j, sigma_j


class JumpModel:
    def __init__(self, T, iterations, N, lam, mu_j, sigma_j):
        self.T = T
        self.iterations = iterations
        self.N = N
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.dt = T/N

    def jump_process(self):
        jump_matrix = np.random.poisson(self.lam * self.dt, (self.N, self.iterations))
        jump_size = np.random.lognormal(self.mu_j, self.sigma_j, (self.N, self.iterations))
        jump_total = (jump_size - 1) * jump_matrix
        return jump_total

    def diffusion_process(self, sigma):
        diffusion_matrix = np.random.normal(0, sigma * np.sqrt(self.dt), (self.N, self.iterations))
        return diffusion_matrix

    def final(self, S0, diffusion, jump, drift):
        S = S0 * np.ones((1, self.iterations))
        random_prices = 1 + diffusion + jump + drift * self.dt
        final = np.vstack((S, random_prices)).cumprod(axis=0)
        return final


class OptionsPrice:
    def __init__(self, final_prices, strike, risk_free, T, option_type):
        self.final_prices = final_prices
        self.strike = strike
        self.risk_free = risk_free
        self.T = T
        self.options_type = option_type

    def options_price(self):
        if self.options_type == 'Call':
            payoffs = np.maximum(self.final_prices[-1] - self.strike, 0)
        elif self.options_type == 'Put':
            payoffs = np.maximum(self.strike - self.final_prices[-1], 0)
        else:
            raise ValueError("Please input 'Call' or 'Put' ")

        avg_payoff = np.mean(payoffs)
        discount_factor = np.exp(-self.risk_free * self.T)
        pv = avg_payoff * discount_factor
        option_price = round(pv, 2)
        return option_price

class Simulation:
    def __init__(self, stock, days, droplist, T, iterations, N, strike, options_type):
        self.stock = stock
        self.days = days
        self.droplist = droplist
        self.T = T
        self.iterations = iterations
        self.N = N
        self.K = strike
        self.options_type = options_type

    def run_simulation(self):
        prices = DataFetch(stock= self.stock, days= self.days, droplist= self.droplist).data_pull()
        S0 = prices.iloc[-1, 0]
        rf = DataFetch.risk_free()

        jump = JumpDiffusion(prices= prices, rf= rf)
        sigma = jump.historical_vol(prices['Close'])
        drift = jump.drift(prices['Close'])
        lam, mu_j, sigma_j = jump.jump_parameters(prices)

        jump_model = JumpModel(T= self.T, iterations= self.iterations, N= self.N, lam= lam, mu_j= mu_j, sigma_j= sigma_j)
        jump_total = jump_model.jump_process()
        diffusion_total = jump_model.diffusion_process(sigma)
        sample_paths = jump_model.final(S0, diffusion= diffusion_total, jump= jump_total, drift= drift)

        options_pricer = OptionsPrice(final_prices= sample_paths,strike= self.K, risk_free= rf, T= self.T, option_type= self.options_type)
        options_ev = options_pricer.options_price()
        return options_ev, sigma, sample_paths

def sample_plot(paths, num_paths=10):
    plot_data = pd.DataFrame(paths)
    plot_sample = plot_data.iloc[:, :num_paths]
    sns.set(style= "dark")
    plt.figure(figsize=(14,8))
    sns.lineplot(data= plot_sample, palette="mako", linewidth= 2)
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.title('Sample Paths')
    st.pyplot(plt.gcf())

#Steamlit#
st.title("Jump-Diffusion Options Pricing Model")

#Sidebar Settings Menu#
with st.sidebar:
    st.title("Model Inputs:")
    #Inputs#
    ticker = st.text_input("Stock Ticker", value= 'SPY')
    option_type = st.selectbox("Option Type", ("Call", "Put"))
    K = st.number_input("Strike Price", value= 550)
    N = st.number_input("Days to Expiration", value= 90)
    T = N/365
    days = st.number_input("Lookback Days for Historical Volatility", value= 252)
    iterations = st.number_input("Number of Simulations", value = 50000)
    drop_list = ['Open', 'High', 'Low', 'Stock Splits', 'Capital Gains', 'Dividends', 'Volume']

    #Run Button
    button = st.button("Run Simulation")

    st.title("Personal Information")
    linkedin_url = 'https://www.linkedin.com/in/michaelwojciechowski93'
    st.markdown(
        f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Michael Wojciechowski</a>',
        unsafe_allow_html=True)

#Run and Streamlit Main Page
if button == True:
    simulation = Simulation(stock= ticker, days= days, droplist= drop_list, T= T, iterations= iterations, N= N, strike= K, options_type= option_type)
    options_ev, vol, sample_paths = simulation.run_simulation()

    col1, col2 = st.columns([1,1])
    col1.metric(label='Historical Volatility', value = f"{round(vol * 100, 2)}%")
    col2.metric(label='Options Theoretical Value', value= f"${round(options_ev, 2)}")

    st.write(f"Graph: 10 of {iterations} Sample Paths Utilized to Simulate Random Price Movements")
    sample_plot(paths= sample_paths)

#Model Description
st.markdown('#### Model Description')
with st.expander("Click Here To Read More About The Model:"):
    st.write("This model for pricing options is a jump-diffusion model for pricing options. To gain a more accurate valuation of options over other monte-carlo based valuation methods, this model incorporates both a jump process (sudden abrupt price changes, similar to interest rates) and a diffusion process (similar to the tempature). 
The model utilizes yfinance to collect hisotrical data for the stock over the specified period of days, as well as the current risk-free intest rate. From the historical data, the historical realized volatility and drift coefficent are calcualted. The model then performs a lookback function over the historical dataset to determine instances where large, abrupt changes in the price of the dataset have occured (2 or greater standard deviations). The frequency of these large jumps is utilzied to as a lambda coefficent, which is fed into a poisson distribution to randomize jump arivials in the model. The mean and standard deviation of these jumps are also calculated, and used as parameters in a random log-normal distribution to generate the size of the jumps.
Log-normal returns are also assumed for the diffusion process. Geometric Brownian Motion is used to generate random stock price movements, which when combined with the jump process form a collective of sample-paths. The sample paths are gnereated out to the date of the options expiration, at which the final prices are collected. The options payoff at these prices are calculated, and discounted to present value to determine the options worth.")
