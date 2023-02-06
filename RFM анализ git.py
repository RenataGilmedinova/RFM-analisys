#!/usr/bin/env python
# coding: utf-8

# # RFM-анализ
# сегментация клиентов по частоте и сумме покупок и выявление тех клиентов, которые приносят больше денег в разрезе направлений

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime as dt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
import plotly.graph_objs as go
import plotly.express as px


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


#Дата2
data2 = pd.read_csv(r'C:\CDP\optidata.csv',encoding='cp1251', sep= ',')


# In[4]:


#Переведем столбец с ном.телефона в строковый формат, для того чтобы убрать первую цифру для удобства мэппинга
data2['phone'] = data2['phone'].astype('string')


# In[5]:


#Убираю первую цифру для удобства мэппинга
data2['phone'] = data2['phone'].str[1:]


# # __________________________________________________________________
# 
# ### Этап 1. Исcледование данных
# 
# 
# *Номера карт представленны исключительно ФЛ

# In[6]:


data2.head(5)


# In[7]:


data2.info()


# In[8]:


#Изменяю формат колонки с датой в датафреймах, Извлекаем из столбца с датой - месяц заправки
data2['date'] = pd.to_datetime(data2['date'])
data2['month'] = data2['date'].dt.month


# In[9]:


#Перевожу пролив из миллилитры в литры
data2['fuel'] = data2['fuel']/1000

#Перевожу сумму заказа из копеек в рубли
data2['total_sum'] = data2['total_sum']/100

#Перевожу сумму оплаты из копеек в рубли
data2['total_paid'] = data2['total_paid']/100


# In[10]:


#виды НП - выгрузка
data2['type_'].unique()


# In[11]:


#Преобразование Номенклатурных групп
data2['type_'] = data2['type_'].apply(lambda x: 'ДТ' if 'ДТ' in x.upper() else x)
data2['type_'] = data2['type_'].apply(lambda x: 'АИ-92' if '92' in x.upper() else x)
data2['type_'] = data2['type_'].apply(lambda x: 'АИ-95' if '95' in x.upper() else x)
data2['type_'] = data2['type_'].apply(lambda x: 'АИ-100' if '100' in x.upper() else x)
data2['type_'] = data2['type_'].apply(lambda x: 'СУГ' if 'ГАЗ' in x.upper() else x)


# In[12]:


#виды НП - выгрузка после преобразования Номенклатурных групп
data2['type_'].unique()


# In[13]:


#Проверка нулевых значений. Такие отсутствуют
data2.isnull().sum()


# In[14]:


#Проверка количества месяцев в датафреме
month_opti = data2.month.unique()

month_opti = len(set(month_opti))
month_opti


# In[15]:


data2


# In[16]:


# убираю фрод
client = pd.pivot_table(data2,
            index=["phone"],
            values=["fuel"],
            aggfunc=[ len])
client = client.droplevel(0, axis=1).rename_axis(None, axis=1).reset_index()
client.columns = ['phone','frequency']


# In[17]:


# удаляю номера, клиенты которых заправляются больше 365 раз в год
data2_new=data2.drop(data2[(data2['phone'].isin(client[client['frequency'] > 365]['phone']))]['phone'].index)


# # __________
# # RFM
# 

# In[18]:


#Дата последнего дня, за который имеем данные
snapshot_date  = dt.datetime(2022,12,31)


# In[19]:


#Создаём таблицу RFM
#R - время от последней заправок пользователя до текущей даты (31/12/22)
#F - суммарное количество заправок у пользователя за всё время 
#M - стоимость заправок за всё время

rfm_data2=data2_new.groupby('phone').agg({'date': lambda x: (snapshot_date  - x.max()).days, # Recency
                                                'sale_id': lambda x: len(x), # Frequency
                                                'total_sum': lambda x: x.sum()})    # Monetary 


# In[20]:


#формат числовых столбцов
pd.options.display.float_format ='{: .3f}'.format


# In[21]:


#переименовываю колонки
rfm_data2.rename(columns={'date':'Recency', 'sale_id':'Frequency', 'total_sum':'MonetaryValue'},inplace = True)
rfm_data2.reset_index()


# In[22]:


# О группах: делим на 3 группы с помощью qcut
#qcut описывается как "функция дискретизации на основе квантилей". 
#Это означает, что qcut делит базовые данные на интервалы равного размера. 
#Функция определяет интервалы с использованием процентилей на основе распределения данных, 
#а не фактических числовых границ интервалов.


# In[23]:


#Создаём таблицу RFM
#R - время от последней заправок пользователя до текущей даты 
#F - суммарное количество заправок у пользователя за всё время 
#M - стоимость заправок за всё время


# In[24]:


# Функция прописывает принадлежность для групп 1,2,3 в зависимости от давности заправки
def f(rfm_data2):
    if rfm_data2['Recency'] <= 30:
        val = '3'
    elif 30 < rfm_data2['Recency'] <= 150:
        val = '2'
    else :
        val = '1'
    return val


rfm_data2['R'] = rfm_data2.apply (f, axis=1)


# In[25]:


#rfm_data2["R"] = pd.qcut(rfm_data2["Recency"], q = 3, labels = [3,2,1]) #чем меньше,тем лучше
rfm_data2["F"] =  pd.qcut(rfm_data2["Frequency"], q = 3, labels = [1,2,3]) #чем больше,тем лучше
rfm_data2["M"] = pd.qcut(rfm_data2["MonetaryValue"], q = 3, labels = [1,2,3]) #чем больше,тем лучше

rfm_data2["RFMSegment"] = rfm_data2["R"].astype("string") + rfm_data2["F"].astype("string") + rfm_data2["M"].astype("string")
rfm_data2["RFMScore"] = rfm_data2["R"].astype("int") + rfm_data2["F"].astype("int") + rfm_data2["M"].astype("int") 


# In[26]:


rfm_data2 = rfm_data2.reset_index()


# In[27]:


plt.figure(figsize=(10,10))
plt.boxplot(rfm_data2['Frequency'])
plt.ylim(0, 100)

# медиана примерно 12 заездов (оранжевая линия)
# 75% заправок совершаются клиентами с частотой менее 30 раз (верхняя граница ящика)
# 75% заправок совершаются клиентами с частотой более 5 раз (нижняя граница ящика)
# много выбросов


# In[28]:


rfm_data2.describe()


# In[29]:


# Распределение количества клиентов по группам
# Количество клиентов по группам совпадает с количеством уникальных клиентов в датафрейме с учетом сортировки
#rfm_data2_g = rfm_data2.groupby(['RFMSegment']).agg({'RFMScore': ['count']}). reset_index()
#rfm_data2_g.columns = ['RFMSegment','RFMScore']


# In[30]:


# перевожу колонку в int для функции группировки
rfm_data2['RFMSegment'] = rfm_data2['RFMSegment'].astype('int')


# In[31]:


# Пишу условия, которые присвоятся клиенту группу анализа в зависимости от RFM сегментации
conditions = [
    (rfm_data2['RFMSegment'] == 321) | (rfm_data2['RFMSegment'] == 322) | (rfm_data2['RFMSegment'] == 323) | (rfm_data2['RFMSegment'] == 331) | (rfm_data2['RFMSegment'] == 333)| (rfm_data2['RFMSegment'] == 332),
    (rfm_data2['RFMSegment'] == 222) | (rfm_data2['RFMSegment'] == 223),
    (rfm_data2['RFMSegment'] == 131) | (rfm_data2['RFMSegment'] == 132) | (rfm_data2['RFMSegment'] == 133),
    (rfm_data2['RFMSegment'] == 211) | (rfm_data2['RFMSegment'] == 212) | (rfm_data2['RFMSegment'] == 213) | (rfm_data2['RFMSegment'] == 221),
    (rfm_data2['RFMSegment'] == 311) | (rfm_data2['RFMSegment'] == 312) | (rfm_data2['RFMSegment'] == 313),
    (rfm_data2['RFMSegment'] == 111) | (rfm_data2['RFMSegment'] == 112) | (rfm_data2['RFMSegment'] == 113) | (rfm_data2['RFMSegment'] == 121) | (rfm_data2['RFMSegment'] == 122) | (rfm_data2['RFMSegment'] == 123),
    (rfm_data2['RFMSegment'] == 231) | (rfm_data2['RFMSegment'] == 232) | (rfm_data2['RFMSegment'] == 233)]


values = ['Лояльные', 'Требующие внимания', 'В зоне риска', 'Спящие редкие с маленьким чеком', 'Новички', 'Уходящие', 'Лояльные на грани']

rfm_data2['segment'] = np.select(conditions, values)


# In[32]:


rfm_data22 = pd.pivot_table(rfm_data2,
            index=["segment"],
            values=["RFMSegment"],
            aggfunc=[len]).reset_index()


# In[33]:


rfm_data22


# In[34]:


rfm_data22.to_csv (r'C:\CDP\my_data.csv', index= False )


# ### Уходящие

# In[35]:


LOSS_op = rfm_data2[rfm_data2['segment'] == 'Уходящие']


# In[36]:


LOSS_op['Recency'].sort_values()


# In[37]:


LOSS_op['Frequency'].sort_values()


# In[38]:


LOSS_op_p = pd.pivot_table(LOSS_op,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()


# In[39]:


LOSS_op_p.columns = ['segment', 'RFMSegment', 'count']


# In[40]:


LOSS_op_111 = rfm_data2[(rfm_data2['segment'] == 'Уходящие') & (rfm_data2['RFMSegment'] == 111)]


# In[41]:


LOSS_op_111['Frequency'].sort_values()


# In[42]:


LOSS_op_p


# ### Лояльные

# In[43]:


loyal_op = rfm_data2[rfm_data2['segment'] == 'Лояльные']


# In[44]:


loyal_optest = rfm_data2[(rfm_data2['segment'] == 'Лояльные')&(rfm_data2['Frequency'] == 7)]


# In[45]:


loyal_optest


# In[46]:


loyal_op['Frequency'].sort_values()


# In[47]:


loyal_op['Recency'].sort_values()


# In[48]:


loyal_op_p = pd.pivot_table(loyal_op,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()
loyal_op_p.columns = ['segment', 'RFMSegment', 'count']


# In[49]:


loyal_op_p


# ### Новички

# In[50]:


new_op = rfm_data2[rfm_data2['segment'] == 'Новички']


# In[51]:


new_op['Frequency'].sort_values()


# In[52]:


new_op['Recency'].sort_values()


# In[53]:


new_op_p = pd.pivot_table(new_op,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()
new_op_p.columns = ['segment', 'RFMSegment', 'count']


# In[54]:


new_op_p


# ### Спящие редкие

# In[55]:


slip_op = rfm_data2[rfm_data2['segment'] == 'Спящие редкие с маленьким чеком']


# In[56]:


slip_op['Frequency'].sort_values()


# In[57]:


slip_op['Recency'].sort_values()


# In[58]:


slip_op_p = pd.pivot_table(slip_op,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()
slip_op_p.columns = ['segment', 'RFMSegment', 'count']


# In[59]:


slip_op_p


# ### Требующие внимания

# In[60]:


attantion_op = rfm_data2[rfm_data2['segment'] == 'Требующие внимания']

attantion_op_p = pd.pivot_table(attantion_op,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()
attantion_op_p.columns = ['segment', 'RFMSegment', 'count']

attantion_op_p


# In[61]:


attantion_op['Recency'].sort_values()


# In[62]:


attantion_op['Frequency'].sort_values()


# ### Лояльные на грани

# In[63]:


loyal_not = rfm_data2[rfm_data2['segment'] == 'Лояльные на грани']

loyal_not_p = pd.pivot_table(loyal_not,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()
loyal_not_p.columns = ['segment', 'RFMSegment', 'count']

loyal_not_p


# In[64]:


loyal_not['Recency'].sort_values()


# ### В зоне риска

# In[65]:


risk_op = rfm_data2[rfm_data2['segment'] == 'В зоне риска']

risk_op_p = pd.pivot_table(risk_op,
            index=["segment", 'RFMSegment'],
            values=["phone"],
            aggfunc=[len]).reset_index()
risk_op_p.columns = ['segment', 'RFMSegment', 'count']

risk_op_p


# In[66]:


risk_op['Recency'].sort_values()


# In[67]:


rfm_data2


# In[68]:


# проверить по количеству клиентов
plt.figure(figsize=(15,6.5))
sns.set_style('darkgrid')
g = sns.barplot(data=rfm_data2, x='RFMSegment', y='RFMScore', ci=False, palette='viridis_r')

plt.show()


# In[69]:


def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal


# In[70]:


pal_vi = get_color('viridis_r', len(rfm_data2))
pal_plas = get_color('plasma_r', len(rfm_data2))
pal_spec = get_color('Spectral', len(rfm_data2))
pal_hsv = get_color('hsv', len(rfm_data2))


# In[71]:


import plotly.express as px
fig = px.treemap(rfm_data2, path=[px.Constant('segment'), 'segment'],
                 values=rfm_data2['RFMScore'],
                 color=rfm_data2['RFMScore'],
                 color_continuous_scale='Spectral_r',
                 color_continuous_midpoint=np.average(rfm_data2['RFMScore'])
                )
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

