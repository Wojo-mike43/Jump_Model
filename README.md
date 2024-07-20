# Jump_Model
Streamlit App: https://56f8lcs2gth2fogiponnbq.streamlit.app

This is a jump-diffusion model for pricing options. To gain a more accurate valuation of options over other monte-carlo based valuation methods, this model incorporates both a jump process (sudden abrupt price changes, similar to interest rates) and a diffusion process (similar to the tempature). 


The model utilizes yfinance to collect hisotrical data for the stock over the specified period of days, as well as the current risk-free intest rate. From the historical data, the historical realized volatility and drift coefficent are calcualted. The model then performs a lookback function over the historical dataset to determine instances where large, abrupt changes in the price of the dataset have occured (2 or greater standard deviations). The frequency of these large jumps is utilzied to as a lambda coefficent, which is fed into a poisson distribution to randomize jump arivials in the model. The mean and standard deviation of these jumps are also calculated, and used as parameters in a random log-normal distribution to generate the size of the jumps.

Log-normal returns are also assumed for the diffusion process. Geometric Brownian Motion is used to generate random stock price movements, which when combined with the jump process form a collective of sample-paths. The sample paths are gnereated out to the date of the options expiration, at which the final prices are collected. The options payoff at these prices are calculated, and discounted to present value to determine the options worth.
