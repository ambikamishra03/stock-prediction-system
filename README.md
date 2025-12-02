# Stock Market Price Predictor using Supervised Learning

### Aim
The aim of this project is to predict stock prices for the next 10 days using one year of historical market data by applying supervised machine learning techniques. The model analyzes past price patterns to forecast future trends, and the results are visualized by plotting both the actual and predicted stock price curves for comparison. This enables better understanding of stock behavior and supports more informed decision-making.

## Setup Instructions
```
    $ workon myvirtualenv                                  [Optional]
    $ pip install -r requirements.txt
    $ python scripts/Algorithms/regression_models.py <input-dir> <output-dir>
```

Download the Dataset needed for running the code from [here](https://drive.google.com/drive/folders/1gCtNTgAAEdQrmmEnDKqhW2GKL-BIeDtl?usp=sharing).

Read project_instructions.txt for more clarity.

### Methodology 
1. Preprocessing and Cleaning
2. Feature Extraction
3. Twitter Sentiment Analysis and Score
4. Data Normalization
5. Analysis of various supervised learning methods
6. Conclusions

### Research Paper
- [Machine Learning in Stock Price Trend Forecasting. Yuqing Dai, Yuning Zhang](http://cs229.stanford.edu/proj2013/DaiZhang-MachineLearningInStockPriceTrendForecasting.pdf)
- [Stock Market Forecasting Using Machine Learning Algorithms. Shunrong Shen, Haomiao Jiang. Department of Electrical Engineering. Stanford University](http://cs229.stanford.edu/proj2012/ShenJiangZhang-StockMarketForecastingusingMachineLearningAlgorithms.pdf)
- [How can machine learning help stock investment?, Xin Guo](http://cs229.stanford.edu/proj2015/009_report.pdf)


### Datasets used
1. http://www.nasdaq.com/
2. https://in.finance.yahoo.com
3. https://www.google.com/finance


<!-- ### Useful Links 
- **Slides**: http://www.slideshare.net/SharvilKatariya/stock-price-trend-forecasting-using-supervised-learning
- **Video**: https://www.youtube.com/watch?v=z6U0OKGrhy0
- **Report**: https://github.com/scorpionhiccup/StockPricePrediction/blob/master/Report.pdf -->

### References
- [Recurrent Neural Networks - LSTM Models](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [ARIMA Models](http://people.duke.edu/~rnau/411arim.htm)
- https://github.com/dv-lebedev/google-quote-downloader
- [Book Value](http://www.investopedia.com/terms/b/bookvalue.asp)
- http://www.investopedia.com/articles/basics/09/simplified-measuring-interpreting-volatility.asp
- [Volatility](http://www.stock-options-made-easy.com/volatility-index.html)
- https://github.com/dzitkowskik/StockPredictionRNN
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [Theano](http://deeplearning.net/software/theano/)



📈 Stock Price Prediction (ML + Flask + React)

This project predicts future stock prices using Linear Regression.  
Includes ML model, Flask backend, and React frontend.

--------------------------------------------------------------
1️⃣ CLONE THE PROJECT
--------------------------------------------------------------
git clone https://github.com/ambikamishra03/stock-prediction-system.git

cd StockPricePrediction

--------------------------------------------------------------
2️⃣ CREATE & ACTIVATE VIRTUAL ENVIRONMENT
--------------------------------------------------------------
python -m venv venv

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

--------------------------------------------------------------
3️⃣ INSTALL REQUIRED LIBRARIES (ML + Backend)
--------------------------------------------------------------
pip install pandas numpy matplotlib scikit-learn yfinance notebook flask flask-cors
 
To load output file:

python scripts/fetch_stock_data1.py input/symbols.txt output/


--------------------------------------------------------------
4️⃣ RUN JUPYTER NOTEBOOK (OPTIONAL)
--------------------------------------------------------------
jupyter notebook:
(open stock_prediction_new.ipynb)

--------------------------------------------------------------
5️⃣ TRAIN MODEL USING NOTEBOOK
--------------------------------------------------------------
Load CSV, visualize, train Linear Regression, predict next 10 days.

--------------------------------------------------------------
6️⃣ RUN BACKEND (FLASK API)
--------------------------------------------------------------
cd backend

pip install flask flask-cors pandas numpy scikit-learn

python app.py

Flask runs at: http://127.0.0.1:5000

--------------------------------------------------------------
7️⃣ RUN FRONTEND (REACT)
--------------------------------------------------------------
cd ../client

npm install

npm start

React runs at: http://localhost:3000

--------------------------------------------------------------
8️⃣ HOW SYSTEM WORKS
--------------------------------------------------------------
- Upload CSV from React
- React sends to Flask
- Flask trains model + predicts next 10 days
- Returns JSON to frontend
- React shows graphs + predictions

--------------------------------------------------------------
CSV FORMAT REQUIRED
--------------------------------------------------------------
Date, Open, High, Low, Close, Volume


--------------------------------------------------------------
Features
--------------------------------------------------------------
✔ Upload CSV  
✔ Predict next 10 days  
✔ Historical graph  
✔ Future prediction graph  
✔ Flask API + React UI  
✔ Clean charts  

END OF FILE
