# Urban AQI Predictor

A machine learning web application that predicts the **Air Quality Index (AQI)** in real time for urban environments, built entirely end-to-end from raw data collection across multiple public sources, through rigorous cleaning and analysis, to a production-ready Flask web application powered by a **Lasso Regression model achieving ~92% accuracy (R² ≈ 0.92).**

## Live Demo

http://65.0.204.194:5000

## Project Overview

Air pollution is one of the most pressing public health challenges in urban India. This project was built to make AQI prediction accessible to anyone not just researchers — through a clean web interface where a user enters a few real-world environmental readings and gets an instant AQI prediction along with a color-coded health advisory.

The entire pipeline was built from scratch: data was gathered from multiple real-world public sources, cleaned, visualized, modeled across four regression algorithms, and finally served through a production-grade Flask web application deployed on AWS

## Data Collection

Data collection was the most time-intensive part of this project — gathering, cross-referencing, and aligning data from different platforms with different formats, update frequencies, and naming conventions took **significant time and effort** before a single line of analysis could be written.

Data was sourced from the following publicly available platforms:
--> Central Pollution Control Board (CPCB):  Official Indian government air quality monitoring data 

--> World Air Quality Index (AQICN): Global real-time AQI readings across monitoring stations 

--> OpenAQ:  Open-source aggregated air quality data platform 

--> IQAir : Commercial-grade air quality and pollution monitoring. 

--> India Meteorological Department (IMD) : Official weather data — temperature, humidity, wind, rainfall 

--> Delhi Pollution Control Committee (DPCC): Delhi-specific station-level pollution records

--> OpenWeatherMap : Weather API data for supplementary meteorological features 

--> Kaggle — Delhi Air Quality Datasets | Publicly available curated and preprocessed datasets 


All collected data was then cleaned, integrated, standardized into a single dataset before any analysis began.


##  Data Preprocessing & Exploratory Data Analysis (EDA)

Once raw data was collected, a thorough preprocessing and exploration pipeline was applied using pandas, matplotlib, and seaborn.

### Initial Inspection & Cleaning

The dataset was first inspected for structure, data types, and column completeness. Categorical distributions like season and weekend flags were checked to understand the data's temporal spread.

### Step 2 — Outlier Detection via Boxplots

Every numeric column was visualized as a boxplot. Pollutant concentration columns — particularly PM2.5, PM10, NO₂, and CO — showed a **large number of outliers**, which is expected in urban air quality data where pollution spikes occur during crop burning seasons, festival periods, and industrial incidents. These were noted and accounted for in the imputation strategy below.


Missing values were filled using the median of each column rather than the mean. This was a deliberate choice — because of the heavy outliers observed in the boxplots, the mean would have been skewed upward and produced unrealistic fill values. The median is robust to outliers and gives a more representative central value for each feature.

Categorical columns (dtype "O") were skipped entirely as they had no missing values.

### Step 4 — Correlation Analysis

A full correlation heatmap was generated across all numeric features. This served two purposes — understanding which pollutants and weather variables were most predictive of AQI, and identifying multicollinear features that could cause instability in linear models (which later informed the choice of regularized regression over plain Linear Regression).


## ⚙️ Feature Engineering & Scaling

### One-Hot Encoding

The two categorical columns — season (Winter, Spring, Monsoon, Autumn) and station_type (Traffic, Industrial, Residential, Background) — were one-hot encoded. **drop_first=True** was used to avoid the dummy variable trap (perfect multicollinearity), which drops Autumn and Background as the reference categories.

After encoding, the dataset expanded to **61 features** total.

### Train-Test Split

A 75/25 train-test split was used with a fixed random state for reproducibility.

### Standard Scaling

All features were standardized using StandardScaler — each feature was transformed to have a mean of 0 and standard deviation of 1. This is critical for regularized models like Lasso and Ridge, because their penalty terms are sensitive to the magnitude of feature values. The scaler was fitted only on training data and then applied to test data to prevent data leakage.

The fitted scaler was saved as scalar.pkl for reuse at prediction time.

### Polynomial Features (for deeper model comparison)

To more rigorously evaluate model performance and understand where Lasso's regularization truly shines, degree-2 polynomial features were also generated. This dramatically expands the feature space with interaction terms and squared features — a setting where Lasso's ability to zero out irrelevant coefficients becomes especially valuable.


## Model Training & Comparison

Four regression models were trained and evaluated — first on scaled features, then on polynomial features — using cross-validation variants (LassoCV, RidgeCV, ElasticNetCV) to automatically find the best regularization strength.


Each model was evaluated on MAE, MSE, RMSE, and R² score:

--> Linear Regression : No regularization — prone to overfitting with many features 

--> Ridge Regression : L2 penalty shrinks all coefficients but keeps all features 

--> ElasticNet : Mix of L1 + L2 — middle ground between Lasso and Ridge 

--> Lasso Regression : L1 penalty — zeros out irrelevant features entirely 


### Why Lasso Performed Best — And What the Data Proved

After training, the features that Lasso completely eliminated were inspected:

This revealed something important — out of 61 features, Lasso's L1 penalty drove several coefficients to exactly zero, effectively performing automatic feature selection. This is the core reason it outperformed the others:

--> Linear Regression kept all 61 features, including noisy and redundant ones, leading to some overfitting

--> Ridge kept all 61 features too — it just made their weights smaller, but could not eliminate them

--> ElasticNet partially zeroed some features, but its blend of L1 and L2 was less aggressive than pure Lasso for this dataset

--> Lasso with its pure L1 penalty found the true signal by discarding features that don't genuinely predict AQI — producing the cleanest and most generalizable model


LassoCV with 5-fold cross-validation also automatically selected the optimal regularization strength (alpha), meaning the model was not manually tuned but selected its own best configuration from the data.


--->                         Final Lasso accuracy: ~92% (R² ≈ 0.92)

The trained Lasso model was saved as lasso.pkl.


## Flask Web Application

### application.py — Backend Logic

The Flask backend was built with several deliberate decisions:

--> Model loaded at startup, not per request  : both scalar.pkl and lasso.pkl are loaded once when the server starts. This avoids re-reading files from disk on every prediction, keeping response times fast

--> 61 features, 16 user inputs   : the model expects 61 features but asking a user to fill 61 fields is impractical. Only the 16 most meaningful and interpretable features are exposed in the UI. The remaining features are either derived from those 16 using established meteorological approximations (dew point, atmospheric pressure, visibility) or set to neutral background values

-->Server-side one-hot encoding :  the Season and Station Type dropdown values are encoded in Python to exactly match the drop_first=True encoding used during training

--> Input validation  : all 16 user inputs are validated against their real-world expected ranges before the prediction pipeline is triggered. Out-of-range or missing values return a clear, readable error message

--> AQI clamping  :  the raw model output is clamped between 0 and 500, which are the real-world boundaries of the AQI scale, preventing physically impossible results

--> AWS Elastic Beanstalk compatibility  :  the app object is named application (not just app) because AWS EB's Python platform specifically looks for a variable with that name. Gunicorn (the production web server) is declared in requirements.txt so EB installs and invokes it automatically — app.run() is never called in production


### Templates/index.html — Frontend

The frontend is a single dark-themed responsive page built with vanilla HTML and CSS:

--> Input fields are grouped into four logical sections — Location & Conditions, Weather, Urban Factors, and Pollutant Concentrations — so users immediately understand what category each input belongs to

--> Sensible default values are pre-filled for urban and weather fields (traffic density, population density, green cover, etc.) to improve the user experience — users only need to update the fields they actually have readings for

--> The prediction result is displayed with a color-coded AQI number, category label (Good / Moderate / Unhealthy / Hazardous), and a tailored health advisory.

--> Fully responsive across desktop and mobile


##  Dependencies

--> flask : Web framework — routing, templating, request handling 

--> numpy : Numerical computation 

--> pandas : Feature vector construction and data manipulation 

--> scikit-learn : StandardScaler, LassoCV, train-test split, metrics 

--> gunicorn : Production WSGI server — used automatically by AWS Elastic Beanstalk 


Install all at once:

pip install -r requirements.txt


## About the Model Files (scalar.pkl and lasso.pkl)

These two files are saved, trained Python objects — think of them as the "brain" of the application. Without them, the app cannot make predictions.

--> scalar.pkl : the fitted StandardScaler. During training, it memorized the average value and spread of every feature across the training dataset. At prediction time, it applies the exact same transformation to the user's input so the model receives numbers in the same scale it was trained on. If you scaled inputs differently, the model would give nonsensical predictions — this file ensures consistency.

--> lasso.pkl : the trained LassoCV model. It stores the learned coefficient for each of the 61 features (many of which are zero, thanks to Lasso's feature selection). When you call model.predict(), it multiplies your scaled input by these coefficients and returns the predicted AQI value.

Both files are generated at the end of notebooks/Air_quality.ipynb using Python's pickle library and must remain in the project root alongside application.py.



## Author

**Alok Gupta**
B.Tech Student — Guru Gobind Singh Indraprastha University (GGSIPU), Delhi


## License

This project is open source and available under the [MIT License](LICENSE).
