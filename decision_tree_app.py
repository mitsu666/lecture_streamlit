import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# サイドバー TOP
# title
st.sidebar.title('Decison Tree Modeler')
# csv upload
uploaded_file = st.sidebar.file_uploader("Choose a file")

# csv uoload
if uploaded_file is not None: # upload されたら実行
  df = pd.read_csv(uploaded_file)
  st.write(df.describe())
else:
  st.sidebar.info('☝️ Upload a CSV file')

if uploaded_file is not None: # upload されたら実行
  st.sidebar.header('Settings')
  # 説明変数を入力する多肢選択box作成
  options = st.sidebar.multiselect(
      '説明変数を選択してください',
      list(df.columns),
      list(df.columns),key=1)
  # 目的変数を入力する選択box
  options1 = st.sidebar.selectbox(
    '目的変数を選択してください',
    tuple(list(set(df.columns)-set(options))),key=2)

  btn01 = st.sidebar.button('Modeling',key=100)
  if btn01:
    # モデリング  
    clf = DecisionTreeClassifier(max_depth=4,
    min_samples_split=50, min_samples_leaf=30, random_state=1)
    X = df[options] # 説明変数
    y = df[options1] # 目的変数
    clf.fit(X,y)
    st.write(options, options1)

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                   feature_names=options,  
                   class_names=options1,
                   filled=True)
    st.pyplot(fig)