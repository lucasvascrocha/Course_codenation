#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[17]:


black_friday.head()


# In[53]:


black_friday.info()


# In[8]:


black_friday.describe()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[89]:


def q1():
    shape = black_friday.shape
    return shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[86]:


def q2():
    n_women_age = len(black_friday[black_friday['Age'] == '26-35'].loc[black_friday['Gender'] == 'F'])
    return n_women_age


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[84]:


def q3():
    usr_uniq = len(black_friday['User_ID'].unique())
    return usr_uniq


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[85]:


def q4():
    num_types = len(black_friday.dtypes.unique())
    return num_types


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[117]:


def q5():
    perc_null = (len(black_friday) -len(black_friday.dropna())) / len(black_friday)
    return perc_null


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[20]:


def q6():
    col_null = black_friday.isnull().sum().max()  
    return int(col_null)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[50]:


def q7():
    mode = black_friday['Product_Category_3'].mode()
    return float(mode)


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[332]:


def q8():
    v_max = black_friday['Purchase'].max()
    v_min = black_friday['Purchase'].min()

    norm = (black_friday['Purchase'] - v_min) / (v_max - v_min)
    mean_norm = norm.mean()
    return float(mean_norm)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[1]:


def q9():
    v_all = black_friday['Purchase']
    v_mean = black_friday['Purchase'].mean()
    v_std = black_friday['Purchase'].std()
    stand = (v_all - v_mean) / v_std
    n_val_in_interval = stand.apply(lambda x :True if (1 >= x >= -1) else False).sum()
    return int(n_val_in_interval)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[51]:


def q10():
    cat2 = black_friday[black_friday['Product_Category_2'].isnull()].index
    cat3 = black_friday[black_friday['Product_Category_3'].isnull()].index
    for cat2, cat3 in zip(cat2,cat3):
        if cat2 == cat3:
            return True
        else:
            return False
    

