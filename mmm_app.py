import streamlit as st
import pandas as pd
import numpy as np
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
from mamimo.datasets import load_fake_mmm
import matplotlib.pyplot as plt
# from mamimo.carryover import ExponentialCarryover
# from mamimo.saturation import ExponentialSaturation
from mamimo.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from mamimo.time_utils import add_time_features, add_date_indicators
from mamimo.time_utils import PowerTrend
from mamimo.carryover import ExponentialCarryover
from mamimo.saturation import ExponentialSaturation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mamimo.analysis import breakdown
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import convolve2d
import scipy.optimize as optimize
import math

class ExponentialSaturation(BaseEstimator, TransformerMixin):
  def __init__(self, a=1.):
    self.a = a
        
  def fit(self, X, y=None):
    X = check_array(X)
    self._check_n_features(X, reset=True) # from BaseEstimator
    return self

  def transform(self, X):
    check_is_fitted(self)
    X = check_array(X)
    self._check_n_features(X, reset=False) # from BaseEstimator
    return 1 - np.exp(-self.a*X)

class ExponentialCarryover(BaseEstimator, TransformerMixin):
  def __init__(self, strength=0.5, length=1):
    self.strength = strength
    self.length = length

  def fit(self, X, y=None):
    X = check_array(X)
    self._check_n_features(X, reset=True)
    self.sliding_window_ = (
        self.strength ** np.arange(self.length + 1)
    ).reshape(-1, 1)
    return self

  def transform(self, X: np.ndarray):
    check_is_fitted(self)
    X = check_array(X)
    self._check_n_features(X, reset=False)
    convolution = convolve2d(X, self.sliding_window_)
    if self.length > 0:
      convolution = convolution[: -self.length]
    return convolution


# サイドバー TOP
# title
st.sidebar.title('Crosstab Market Mix Modeler')
# csv upload
uploaded_file = st.sidebar.file_uploader("Choose a file")

# uoload されたデータ可視化
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  df['datetime'] = pd.to_datetime(df['datetime'])
  df.set_index('datetime',inplace=True)
  st.subheader('DataFrame')
  st.write(df)
  st.subheader('統計量')
  st.write(df.describe())
  # st.subheader('Info')
  # st.write(df.dtypes)
else:
  st.sidebar.info('☝️ Upload a CSV file')

if uploaded_file is not None:
  st.sidebar.header('Settings')
  options = st.sidebar.multiselect(
      '広告変数を選択してください',
      list(df.columns),
      list(df.columns),key=1)
  # options = st.multiselect(
  #     '広告変数を選択してください',
  #     list(df.columns),
  #     list(df.columns))

  tdic = {}
  for i,col in enumerate(options):
    tdic[col] = st.sidebar.multiselect(
      f'{col}のハイパーパラメーターを選んで下さい',
      ['Carryover', 'Saturation'],
      ['Carryover', 'Saturation'],key=i+2)

  options1 = st.sidebar.multiselect(
    'その他の説明変数を選択してください',
    list(set(df.columns)-set(options)),
    list(set(df.columns)-set(options)),key=10)

  options2 = st.sidebar.multiselect(
    '目的変数を選択してください',
    list(set(df.columns)-set(options)-set(options1)),
    list(set(df.columns)-set(options)-set(options1)),key=11) 

  # st.write(st.session_state['modeling'])
  btn01 = st.sidebar.button('Modeling',key=100)

    # if 'modeling' not in st.session_state:
  #   st.session_state['modeling'] = False 
  if btn01:
    st.session_state['modeling'] = True 
  if st.session_state['modeling']:
    # st.sidebar.write('モデリング実行')
    # st.write(st.session_state['modeling'])
     # 説明可能と目的変数に分離
    X = df[options+options1]
    # st.write(X.shape)
    y = df[options2]
    preprocess = ColumnTransformer(
     [(col+'_pipe', Pipeline([
            ('carryover', ExponentialCarryover()),
            ('saturation', ExponentialSaturation())
     ]), [col]) for col in options],
     remainder = 'passthrough'
    ) 


    model = Pipeline([
    ('preprocess', preprocess),
    ('regression', LinearRegression(
        positive=False,
        fit_intercept=True) # no intercept because of the months
    )
    ])

    # ハイパーパラメータチューニング
    d1 = {'preprocess__'+col+'_pipe__carryover__strength': UniformDistribution(0, 1) 
    for col in options}
    d2 = {'preprocess__'+col+'_pipe__carryover__length': IntUniformDistribution(0, 6) 
    for col in options}
    d3 = {'preprocess__'+col+'_pipe__saturation__a': UniformDistribution(0, 0.01) 
    for col in options}
    d1.update(d2)
    d1.update(d3)
    # st.write(d1)
    tuned_model = OptunaSearchCV(
        estimator=model,
        param_distributions=d1,
        n_trials=10,
        cv=TimeSeriesSplit(),
        random_state=0
    )
    tuned_model.fit(X, y)
    st.subheader('決定係数')
    st.info(tuned_model.score(X, y))
    # st.write(X)
    # 時系列予測値確認
    # st.write(X)
    preds = tuned_model.predict(X)
    # st.write(np.hstack([preds, np.array(y)]))

    chart_data = pd.DataFrame(
    np.hstack([preds, np.array(y)]),
    columns=['Preds', 'Actual'],index=X.index)
    # st.write(chart_data)
    st.subheader('予測実績プロット')
    st.line_chart(chart_data)
    st.subheader('学習済ハイパーパラメータ')
    st.write(pd.DataFrame.from_dict(tuned_model.best_params_, orient='index'))
    st.subheader('回帰係数')
    st.write(pd.DataFrame(
      tuned_model.best_estimator_.named_steps['regression'].coef_,columns=X.columns))



    adstock_data = pd.DataFrame(
        tuned_model.best_estimator_.named_steps['preprocess'].transform(X),
        #columns=X.columns,
        index=X.index
    )
    weights = tuned_model.best_estimator_.named_steps['regression'].coef_

    base = tuned_model.best_estimator_.named_steps['regression'].intercept_ # 切片なし回帰であるため不要
    
    unadj_contributions = adstock_data.mul(weights)
    unadj_contributions['Base'] = base[0]


    adj_contributions = (unadj_contributions
                        .div(np.array(unadj_contributions.sum(axis=1)), axis=0)
                        .mul(np.array(y), axis=0)
                        )
    adj_contributions = pd.concat([adj_contributions.iloc[:,:len(options)],
            adj_contributions.iloc[:,len(options):].sum(axis=1)],axis=1)
    adj_contributions.columns = list(X.columns[:len(options)]) + ['Baseline'] 

    st.area_chart(adj_contributions)

    # ROIを計算する
    st.subheader('ROIを計算')
    for channel in options:
      roi = adj_contributions[channel].sum() / X[channel].sum()
      st.info(f'{channel}: {roi:.4f}')

    
    fig, axs = plt.subplots(nrows=math.floor(len(options) // 2), ncols=2, figsize=(10, 6))
    # plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Saturation effects", fontsize=18, y=0.95)
    for col,ax in zip(options, axs.ravel()):
      max_ = X[col].max()
      # st.write(max_)
      s = tuned_model.best_estimator_.named_steps.preprocess.named_transformers_[col+'_pipe'
      ].named_steps.saturation.fit_transform(np.array(np.linspace(0,max_).reshape(-1,1)))
      t = np.linspace(0,max_)
      ax.set_title(col)
      ax.plot(t, s)

      tmp = np.array(X.mean()[col]).reshape(-1,1)
      tmp2 = tuned_model.best_estimator_.named_steps.preprocess.named_transformers_[col+'_pipe'
      ].named_steps.saturation.fit_transform(tmp)
      ax.scatter(tmp, tmp2, color='red')
      ax.grid()
    st.pyplot(fig)


    slidedic = {}
    for i, col in enumerate(options1):
      slidedic[col] = st.sidebar.number_input(label=f'{col}の値を入力してください', key=col+'form')
    sup = st.sidebar.number_input(label=f'予算上限を入力してください', key='budgetform')
    if st.sidebar.button('Optimization', key=101):
      st.sidebar.write('最適化実行')
      # 目的関数定義
      def func(x):
          X_ = pd.DataFrame(x,index=options).T
          temp = pd.DataFrame({k:[v] for k,v in slidedic.items()})
          X_ = pd.concat([X_, temp],axis=1)
          # st.write(X_)
          return(-1*tuned_model.predict(X_))

      # 制約条件定義
      def con(x):
          return x.sum()-sup
      cons = [{'type':'eq', 'fun': con}] + [{'type':'ineq', 'fun': lambda x:x[i]} for i in range(len(options))]
      
      x0 = np.ones(len(options))*(sup/len(options))

      res = optimize.minimize(func, x0, constraints=cons)
      st.subheader('最適解')
      st.write(res.x)

      st.info(func(x0)*-1)
      st.info(func(res.x)*-1)

