#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required packages from python library
import streamlit as st
import pandas as pd
import pickle
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")


# In[2]:


pd.set_option("display.max_columns", None)


# # Model

# In[3]:


data = pd.read_csv("data.csv").drop("Unnamed: 0", axis=1)


# In[4]:


#data


# In[5]:


X = data.drop(["R", "Ntc Valor", "Nic Valor", "Nvc Valor", "EPC", "TARGET", "Ntc Limite"], axis=1)
y = data[["R", "Ntc Valor", "Nic Valor", "Nvc Valor", "Ntc Limite"]]


# In[6]:


#X.columns


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[8]:


# et_r.fit(X_train, y_train["R"])
# preds = et_r.predict(X_test)
# r2_score(y_test["R"], preds)


# In[9]:


# et_ntc.fit(X_train, y_train["Ntc Valor"])
# preds = et_ntc.predict(X_test)
# r2_score(y_test["Ntc Valor"], preds)


# In[10]:


# et_ntcl.fit(X_train, y_train["Ntc Limite"])
# preds = et_ntcl.predict(X_test)
# r2_score(y_test["Ntc Limite"], preds)


# In[11]:


# et_nic.fit(X_train, y_train["Nic Valor"])
# preds = et_nic.predict(X_test)
# r2_score(y_test["Nic Valor"], preds)


# In[12]:


# et_nvc.fit(X_train, y_train["Nvc Valor"])
# preds = et_nvc.predict(X_test)
# r2_score(y_test["Nvc Valor"], preds)


# # Data

# In[13]:


def period_to_epoch(x):
    if pd.isna(x) == True:
        return "is null"
    if x == "anterior a 1918":
        return 0
    if x == "entre 1919 a 1945":
        return 1
    if x== "entre 1946 a 1960":
        return 2
    if x== "entre 1961 a 1970":
        return 3
    if x== "entre 1971 a 1980":
        return 4
    if x== "entre 1981 a 1990":
        return 5
    if x== "entre 1991 a 1995":
        return 6
    if x== "entre 1996 a 2000":
        return 7
    if x== "entre 2001 a 2005":
        return 8
    else:
        return 9


# In[14]:


def epochs_to_period(x):
    if x == 0:
        return "anterior a 1918"
    if x == 1:
        return "entre 1919 a 1945"
    if x== 2:
        return "entre 1946 a 1960"
    if x== 3:
        return "entre 1961 a 1970"
    if x== 4:
        return "entre 1971 a 1980"
    if x== 5:
        return "entre 1981 a 1990"
    if x== 6:
        return "entre 1991 a 1995"
    if x== 7:
        return "entre 1996 a 2000"
    if x== 8:
        return "entre 2001 a 2005"
    else:
        return "Posterior a 2005"


# In[15]:


period_df = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
period_df["label"] = period_df[0].apply(epochs_to_period)


# In[16]:


#period_df


# In[17]:


typology_type = ['> T6', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']


# In[18]:


typology_labels = [0, 1, 2, 3, 4, 5, 6, 7]


# In[19]:


typology_df = pd.DataFrame([typology_type, typology_labels]).T.apply(np.roll, shift=-1)


# In[20]:


#typology_df


# In[21]:


epc_type = ['Edif??cio', 'Fra????o (s/ PH e com utiliza????o independente)', 'Fra????o Aut??noma (com PH constitu??da)']
epc_type_labels = [0,1, 2]
epc_type_df = pd.DataFrame([epc_type, epc_type_labels]).T


# In[22]:


district_types = pd.read_csv("disctrict_types.csv")


# In[184]:


wall_types = pd.read_csv("wall_types.csv")
roof_types = pd.read_csv("roof_types.csv")
floor_types = pd.read_csv("floors_types.csv")
window_types = pd.read_csv("window_types.csv")


# In[185]:


ac_sources = pd.read_csv("ac_sources.csv").iloc[:12]
ac_types = pd.read_csv("ac_types.csv").iloc[:16]

dhw_sources = pd.read_csv("dhw_sources.csv")
dhw_types = pd.read_csv("dhw_types.csv")


# ## Walls

# In[25]:


#wall_types


# In[26]:


epoch_walls = data.groupby("epoch").mean()["walls_type"].astype("int")


# In[27]:


#epoch_walls


# In[28]:


def period_to_wall(x):
    for wall in enumerate(epoch_walls):
        if period_to_epoch(x) == wall[0]:
            return wall[1]
            


# In[29]:


#period_to_wall("entre 2001 a 2005")


# ## ROOFS

# In[30]:


#roof_types


# In[31]:


epoch_roofs = data.groupby("epoch").mean()["roofs_type"].astype("int")


# In[32]:


#epoch_roofs


# In[33]:


def period_to_roof(x):
    for roof in enumerate(epoch_roofs):
        if period_to_epoch(x) == roof[0]:
            return roof[1]


# In[34]:


#period_to_roof("entre 2001 a 2005")


# ## Floors

# In[35]:


#floor_types["solution"]


# In[36]:


epoch_floors = data.groupby("epoch").mean()["floors_type"].astype("int")


# In[37]:


#epoch_floors


# In[38]:


def period_to_floor(x):
    for floor in enumerate(epoch_floors):
        if period_to_epoch(x) == floor[0]:
            return floor[1]


# In[39]:


#period_to_floor("entre 2001 a 2005")


# ## Windows

# In[40]:


#window_types


# In[41]:


epoch_windows = data.groupby("epoch").mean()["window_type"].astype("int")


# In[42]:


#epoch_windows


# In[43]:


def period_to_window(x):
    for window in enumerate(epoch_windows):
        if period_to_epoch(x) == window[0]:
            return  window[1]


# In[44]:


#period_to_window("entre 2001 a 2005")


# In[45]:


#ac_sources["0"]


# In[46]:


#ac_types["0"]


# In[47]:


#dhw_types


# In[48]:


#dhw_sources["0"]


# # Interface

# In[49]:


st.write("""
# Ferramenta de Reabilita????o de Im??veis e Edif??cios

Esta app prev?? o o seu certificado energ??tico, e optimiza as reabilita????es ??ptimas que pode efectuar dentro do seu limite de or??amento.""")
st.write("---")


# In[50]:


#ADJUST COLUMN WIDTH

st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
width: 600px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
width: 600px;
margin-left: -600px;
}
</style>
""",
unsafe_allow_html=True
)


# In[51]:


# Sidebar
# Header of Specify Input Parameters
st.header('Aqui pode especificar os detalhes do seu apartamento ou Edif??cio ')

def user_base_input():
    st.subheader("Detalhes gerais")
    
    District = st.selectbox("Localiza????o do seu im??vel", district_types["1"], index=6)
    B_type = st.selectbox("Tipo de Certificado", 
                                  ["Edif??cio", "Propriedade Horizontal"], 
                                  index=1)
    if B_type == "Propriedade Horizontal":
        floor_position = st.selectbox("Localiza????o piso do seu im??vel", 
                                              ["Piso T??rreo", "Piso interm??dio", "??ltimo Piso"], 
                                              index=2)
        N_floors = st.number_input("N??mero total de pisos do edif??cio",
                                           step=1, 
                                           value=2)
    else:
        N_floors = st.number_input('N??mero de pisos do seu Edif??cio', 
                                           step=1, value=5)
        floor_position = 0

    Period = st.selectbox('Periodo de Constru????o', period_df["label"], index=7)
    f_area = st.number_input('??rea ??til',value=100,  step=1)
    f_height = st.number_input('P?? direito', value=2.80)
    typology = st.selectbox('Assoalhadas', typology_df[0], index=3)

    st.subheader("Detalhes de equipamentos de climatiza????o")

    ac_type = st.selectbox("Tipo de Equipamento de climatiza????o no seu Im??vel ou Edif??cio", 
                           ac_types["0"].append(pd.Series("Nao possuo equipamentos de climatiza????o"), 
                                                ignore_index=True), 
                           index=5) #16 - nao possuo
    
    if ac_type != "Nao possuo equipamentos de climatiza????o":
        ac_source = st.selectbox("Tipo de Fonte de Energia para climatiza????o", ac_sources["0"], 
                                         index=4)
        nr_ac_units = st.number_input("N??mero de equipamentos para climatiza????o no seu im??vel ou edif??cio", 
                                      value=1, 
                                      step=1,
                                      min_value=1)
    else:
        ac_source = -1
        nr_ac_units = -1

    st.subheader("Detalhes de equipamentos de ??guas Quentes e Sanit??rias")

    dhw_type = st.selectbox('Tipo de Equipamento AQS no seu Im??vel ou Edif??cio', 
                            dhw_types["0"].append(pd.Series("Nao possuo equipamentos de AQS"), 
                                                  ignore_index=True), 
                            index= 8)
    if dhw_type != "Nao possuo equipamentos de AQS":
        dhw_source = st.selectbox("Tipo de Fonte de Energia para AQS", dhw_sources["0"], 
                                         index=4)
        nr_dhw_units = st.number_input("N??mero de equipamentos para AQS no seu im??vel ou edif??cio.", 
                                               value=1,
                                               step=1,
                                               min_value=1)
    else:
        dhw_source = -1
        nr_dhw_units = -1

    df = pd.DataFrame([District,
                      B_type, 
                      floor_position, 
                      N_floors, 
                      Period, 
                      f_area, 
                      f_height, 
                      typology,
                      ac_type,
                      ac_source,
                      nr_ac_units,
                      dhw_type, 
                      dhw_source,
                      nr_dhw_units]).T
    df.columns = [ "District",
                 "B_type", 
                 "floor_position", 
                 "N_floors", 
                 "Period", 
                 "f_area", 
                 "f_height", 
                 "typology",
                 "ac_type",
                 "ac_source",
                 "nr_ac_units",
                 "dhw_type", 
                 "dhw_source",
                 "nr_dhw_units"]
    return df


# In[52]:


base_inputs = user_base_input()
#base_inputs


# In[53]:


# st.sidebar.write("---")
# st.sidebar.caption("Para proceder com a previs??o do seu certificado energ??tico precisamos de dados um pouco mais detalhados. No entanto, podemos simplificar estes dados com base no ano de constru????o do seu im??vel ou edif??cio, e as solu????es construtivas t??picas para essa ??poca de constru????o.")
# st.sidebar.caption("Este processo pode gerar informa????es incorrectas particularmente se o seu im??vel ou ed??ficio j?? sofreu obras de reabilita????o, aumentando assim o erro m??dio da previs??o.")


# In[54]:


st.write("---")
st.caption("Para proceder com a previs??o do seu certificado energ??tico precisamos de dados um pouco mais detalhados. No entanto, podemos simplificar estes dados com base no ano de constru????o do seu im??vel ou edif??cio, e as solu????es construtivas t??picas para essa ??poca de constru????o.")
st.caption("Este processo pode gerar informa????es incorrectas particularmente se o seu im??vel ou ed??ficio j?? sofreu obras de reabilita????o, aumentando assim o erro m??dio da previs??o.")


# In[55]:


# df1 = proceeder()
# df1


# In[56]:


#floor_types


# In[57]:


def user_advanced_inputs():
    with st.expander("Detalhes Avan??ados"):
        base_wall_type = period_to_wall(base_inputs["Period"].iloc[0])
        base_roof_type = 5#period_to_roof(base_inputs["Period"].iloc[0])
        base_floor_type = 8#period_to_floor(base_inputs["Period"].iloc[0])
        base_window_type = period_to_window(base_inputs["Period"].iloc[0])
        base_wall_area = np.sqrt(base_inputs["f_area"].iloc[0])*base_inputs["f_height"].iloc[0]*2
        
        st.subheader("Detalhes Paredes")
        #WALLS        
        wall_type = st.selectbox("Tipo de solu????o construtiva das suas paredes", 
                                 wall_types["Solution"], 
                                 index= base_wall_type)
        wall_area = st.number_input("??rea das Paredes Exteriores", 
                                    value= base_wall_area)
        #ROOF
        st.subheader("Detalhes Cobertura")
        
        
        if base_inputs["B_type"].iloc[0] == "Edif??cio": #B_type
            roof_type = st.selectbox("Tipo de Cobertura do seu seu Im??vel ou Edif??cio:", 
                                     roof_types["Solution"],
                                     index= base_roof_type)
        elif base_inputs["floor_position"].iloc[0] == "??ltimo Piso": #floor_position
            roof_type = st.selectbox("Tipo de Cobertura do seu seu Im??vel ou Edif??cio:", 
                                     roof_types["Solution"],
                                     index=base_roof_type)
        else:
            roof_type = -1
            st.caption("O seu Im??vel ou Edif??cio n??o necessita destas informa????es.")
            
        if roof_type != -1:
            roof_area = st.number_input("??rea da Cobertura", 
                                        value= base_inputs["f_area"].iloc[0])
        else:
            roof_area = -1
            
        st.subheader("Detalhes Pavimento")
        #FLOORS
        if base_inputs["B_type"].iloc[0] == "Edif??cio": #B_type
            floor_type = st.selectbox("Tipo de Pavimento do seu seu Im??vel ou Edif??cio:", 
                                      floor_types["solution"], 
                                      index=base_floor_type)
        elif base_inputs["floor_position"].iloc[0] == "Piso T??rreo": #floor_position
            floor_type = st.selectbox("Tipo de Pavimento do seu seu Im??vel ou Edif??cio:", 
                                     floor_types["solution"], 
                                     index=base_floor_type)
        else:
            floor_type = -1
            st.caption("O seu Im??vel ou Edif??cio n??o necessita destas informa????es.")
            
        if floor_type!= -1:
            floor_area = st.number_input("??rea de Pavimento", 
                                        value= base_inputs["f_area"].iloc[0])
            
        else:
            floor_area=-1
            

        st.subheader("Detalhes de Envidra??ados")
        window_area = st.number_input("??rea de Envidra??ados", value=wall_area*0.2)
        window_type = st.selectbox("Tipo de Solu????o construtiva das janelas no seu Im??vel ou Edif??cio", 
                                   window_types["Tipo de Solu????o 1"],
                                   index=base_window_type)


    df2 = pd.DataFrame([wall_type, wall_area, roof_type, roof_area, floor_type, floor_area, window_type, window_area]).T #, ac_type, nr_ac_units, dhw_type, nr_dhw_units]).T
    df2.columns = ["wall_type", "wall_area", "roof_type", "roof_area", "floor_type", "floor_area", "window_type", "window_area"] #, "ac_type", "nr_ac_units", "dhw_type", "nr_dhw_units"]
    return df2


# In[58]:


advanced_inputs = user_advanced_inputs()
#advanced_inputs


# In[59]:


full_user_data = pd.concat([base_inputs, advanced_inputs],axis=1)


# In[60]:


#full_user_data


# In[61]:


def district_to_int(x):
    for i in enumerate(district_types["1"]):
        if x == i[1]:
            return i[0]

def tipo_int(x):
    if x== "Propriedade Horizontal":
        tipo = 2
    else:
        tipo=0
    return tipo

def assoa_int(x):
    for i in enumerate(typology_df[0]):
        if x== i[1]:
            return i[0]
        
def wall_type_to_u(x):
    for i in enumerate(wall_types["Solution"]):
        if x == i[1]:
            return wall_types["Coeficiente Solu????o"].iloc[i[0]]

def wall_type_to_int(x):
    for i in enumerate(wall_types["Solution"]):
        if x == i[1]:
            return i[0]

def roof_type_to_u(x):
    for i in enumerate(roof_types["Solution"]):
        if x ==-1:
            return -1
        elif x == i[1]:
            return roof_types["Coeficiente Solu????o"].iloc[i[0]]

def roof_type_to_int(x):
    for i in enumerate(roof_types["Solution"]):
        if x ==-1:
            return -1
        elif x == i[1]:
            return i[0]

def floor_type_to_u(x):
    for i in enumerate(floor_types["solution"]):
        if x ==-1:
            return -1
        elif x == i[1]:
            return floor_types["Coeficiente Solu????o"].iloc[i[0]]

def floor_type_to_int(x):
    for i in enumerate(floor_types["solution"]):
        if x ==-1:
            return -1
        elif x == i[1]:
            return i[0]

def window_type_to_u(x):
    for i in enumerate(window_types["Tipo de Solu????o 1"]):
        if x == i[1]:
            return window_types["Uwdn"].iloc[i[0]]

def window_type_to_int(x):
    for i in enumerate(window_types["Tipo de Solu????o 1"]):
        if x == i[1]:
            return i[0]
        
def ac_source_to_int(x):
    for i in enumerate(ac_sources["0"]):
        if x == -1:
            return -1
        elif x == i[1]:
            return i[0]
    
def ac_type_to_int(x):
    for i in enumerate(ac_types["0"]):
        if x == "Nao possuo equipamentos de climatiza????o":
            return -1
        elif x == i[1]:
            return i[0]

def dhw_source_to_int(x):
    for i in enumerate(dhw_sources["0"]):
        if x == -1:
            return -1
        elif x == i[1]:
            return i[0]
    
def dhw_type_to_int(x):
    for i in enumerate(dhw_types["0"]):
        if x == "Nao possuo equipamentos de AQS":
            return -1
        elif x == i[1]:
            return i[0]


# In[62]:


model_inputs = pd.DataFrame(np.repeat(0, 25)).T
model_inputs.columns = X_train.columns


# In[63]:


model_inputs["Distrito"] = district_to_int(full_user_data["District"].iloc[0])
model_inputs["Tipo de Im??vel"] = tipo_int(full_user_data["B_type"].iloc[0])
model_inputs["??rea ??til de Pavimento"] = full_user_data["f_area"].iloc[0]
model_inputs["P?? Direito M??dio"] = full_user_data["f_height"].iloc[0]
model_inputs["Tipologia"] = assoa_int(full_user_data["typology"].iloc[0])
model_inputs["N??mero Total de Pisos"] = full_user_data["N_floors"]
model_inputs["epoch"] = period_to_epoch(full_user_data["Period"].iloc[0])
model_inputs["walls_u"] = wall_type_to_u(full_user_data["wall_type"].iloc[0])
model_inputs["walls_area"] = full_user_data["wall_area"]
model_inputs["walls_type"] = wall_type_to_int(full_user_data["wall_type"].iloc[0])
model_inputs["roofs_u"] = roof_type_to_u(full_user_data["roof_type"].iloc[0])
model_inputs["roofs_area"] = full_user_data["roof_area"]
model_inputs["roofs_type"] = roof_type_to_int(full_user_data["roof_type"].iloc[0])
model_inputs["floors_area"] = full_user_data["floor_area"]
model_inputs["floors_u"] = floor_type_to_u(full_user_data["floor_type"].iloc[0])
model_inputs["floors_type"] = floor_type_to_int(full_user_data["floor_type"].iloc[0])
model_inputs["window_area"] = full_user_data["window_area"]
model_inputs["window_u"] = window_type_to_u(full_user_data["window_type"].iloc[0])
model_inputs["window_type"] = window_type_to_int(full_user_data["window_type"].iloc[0])
model_inputs["ac_source"] = ac_source_to_int(full_user_data["ac_source"].iloc[0])
model_inputs["ac_equipment"] = ac_type_to_int(full_user_data["ac_type"].iloc[0])
model_inputs["nr_ac_units"] = full_user_data["nr_ac_units"]
model_inputs["dhw_source"] = dhw_source_to_int(full_user_data["dhw_source"].iloc[0])
model_inputs["dhw_equipment"] = dhw_type_to_int(full_user_data["dhw_type"].iloc[0])
model_inputs["nr_dhw_units"] = full_user_data["nr_dhw_units"]


# In[64]:


#model_inputs


# In[65]:


# user_view_inputs = full_user_data.copy()
# user_view_inputs.columns =  ["Tipo", 
#                              "Posi????o do piso", 
#                              "N??mero de pisos",
#                              "Per??odo de constru????o",
#                              "??rea ??til",
#                              "P?? direito",
#                              "Assoalhadas",
#                              "Tipo de Climatiza????o",
#                              "Fonte energia Climatiza????o",
#                              "N??mero de unidades para Aquecimento",
#                              "Tipo de AQS",
#                              "Fonte energia AQS",
#                              "N??mero de equipamentos para AQS",
#                              "Tipo de paredes",
#                              "??rea das paredes",
#                              "Tipo de Cobertura",
#                              "??rea de Cobertura",
#                              "Tipo de Pavimento",
#                              "??rea de Pavimento",
#                              "Tipo de Envidra??ados",
#                              "??rea de Envidra??ados"]

# st.write("Aqui pode consultar os dados usados para a previs??o do seu certificado:")
# st.dataframe(user_view_inputs.T.astype("str"))


# # Model Generation

# In[66]:


r_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
ntc_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
nic_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
nvc_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)


# In[1]:


# with st.spinner("""O c??lculo do seu certificado n??o substitui a avalia????o realizada por um perito.
#                 As informa????es aqui avan??adas representam uma aproxima????o ao c??lculo do certificado energ??tico 
#                 com um erro m??dio de uma classe energ??tica."""):
# r_model.fit(X_train, y_train["R"])
# ntc_model.fit(X_train, y_train["Ntc Valor"])
# nic_model.fit(X_train, y_train["Nic Valor"])
# nvc_model.fit(X_train, y_train["Nvc Valor"])


# In[68]:


# @st.cache(allow_output_mutation=True)  # ???? Added this
# def r_():
#     return r_model.fit(X_train, y_train["R"])

# @st.cache(allow_output_mutation=True)  # ???? Added this
# def ntc_():
#     return ntc_model.fit(X_train, y_train["Ntc Valor"])


# In[69]:


col_a, col_c, colb = st.columns(3)
simulate_button = col_c.button('Simule Aqui')
#if simulate_button:
with st.spinner("""O c??lculo do seu certificado n??o substitui a avalia????o realizada por um perito.
                As informa????es aqui avan??adas representam uma aproxima????o ao c??lculo do certificado energ??tico 
                com um erro m??dio de uma classe energ??tica."""):
    @st.cache(allow_output_mutation=True)  # ???? Added this
    def r_():
        return r_model.fit(X_train, y_train["R"])


    @st.cache(allow_output_mutation=True)  # ???? Added this
    def ntc_():
        return ntc_model.fit(X_train, y_train["Ntc Valor"])

    @st.cache(allow_output_mutation=True)  # ???? Added this
    def nvc_():
        return nvc_model.fit(X_train, y_train["Nvc Valor"])

    @st.cache(allow_output_mutation=True)  # ???? Added this
    def nic_():
        return nic_model.fit(X_train, y_train["Nic Valor"])

et_r = r_() 
et_ntc =  ntc_()
et_nvc =  nvc_() 
et_nic =  nic_() 

# else:
#     mse_nvc = 0
#     mse_nic = 0
#     mse_ntc = 0
#     mse_r = 0
#     st.write("")
#     st.header('Carregue no bot??o da barra lateral para iniciar a previs??o do seu Certificado Energ??tico')


# import seaborn as sns

# sns.histplot(y_test["R"])

#  r_pal = sns.light_palette("#a6ad1e", reverse=False, as_cmap=True)

# r_pal

# errors_r = 100*(et_r.predict(X_test)- y_test["R"])/et_r.predict(X_test)

# test_set = pd.concat([errors_r, y_test["R"]], axis=1)

# test_set.columns = ["error [%]", "R [ratio]"]

# r2_score(y_test["R"], et_r.predict(X_test))

# g = sns.jointplot(x="error [%]", y="R [ratio]",data=test_set[(test_set["R [ratio]"] < 3) & 
#                                                                  (test_set["error [%]"] < 100) &
#                                                                  (test_set["error [%]"] > -100)], kind='kde', fill=True, space=0, color="Green", cmap="Greens")

# In[70]:


def r_to_epc_fig(r):
    if r <= 0.25:
        return "epcs/A+.png"
    elif r <= 0.50:
        return "epcs/A.png"
    elif r <= 0.75:
        return "epcs/B.png"
    elif r <= 1.00:
        return "epcs/B-.png"
    elif r <= 1.50:
        return "epcs/C.png"
    elif r <= 2.00:
        return "epcs/D.png"
    elif r <= 2.50:
        return "epcs/E.png"
    else:
        return "epcs/F.png"


# In[71]:


area_calc = model_inputs["??rea ??til de Pavimento"].iloc[0]


# In[72]:


if simulate_button:

    r = np.round(et_r.predict(model_inputs)[0], 2)
    ntc = np.round(et_ntc.predict(model_inputs)[0]*area_calc,0)
    nic = np.round(et_nic.predict(model_inputs)[0]*area_calc, 0)
    nvc = np.round(et_nvc.predict(model_inputs)[0]*area_calc,0)
    
    col_image1, col_image_c, col_image2 = st.columns(3)
    col_image_c.image(r_to_epc_fig(r))
    
    col1, col2, col3 = st.columns(3)

    
    col1.image("epcs/cooling.png", width=35)
    col1.metric("Energia para arrefecimento (kWh/ano)", int(nvc))
    col2.image("epcs/heating.png", width=35)
    col2.metric("Energia para aquecimento (kWh/ano)", int(nic))
    col3.image("epcs/en.png", width=35)
    col3.metric("Energia total anual (kWh/ano)", int(ntc))


# # Optimization

# In[73]:


st.write("---")
st.write("""Os certificados energ??ticos categorizam os im??veis ou edif??cios de acordo com a sua efici??ncia energ??tica,
             de maneira a que se identifique e promova a reabilita????o dos casos mais graves. Como tal, o governo aprovou
             um pacote de medidas que visa incentivar essa reabilita????o: O "Fundo de comparticipa????o ambiental".
          """)

st.write("""
         O fundo de comparticipa????o ambiental cobre certas despesas de reabilita????o de edif??cios, conforme a medida 
         de reabilita????o aplicada, de um conjunto total de medidas definidas pelo governo tais como: a aplica????o de pain??is solares, 
         isolamento de coberturas e paredes, mudan??a de envidra??ados, entre outros.
          """)

st.write("""
         Adicionalmente, outro incentivo governamental ?? o desconto ou isen????o de IMI mediante certos casos de
         certifica????o energ??tica que melhoram significativamente ap??s a reabilita????o.
          """)

st.write("""
         Com vista nestas informa????es, caso queira efectuar interven????es de reabilita????o no seu im??vel e/ou edif??cio, 
         pode usar a nossa ferramenta para, com base no seu or??amento, sugerir combina????es ??ptimas de interven????es referidas
         no fundo de comparticipa????o ambiental, para minimizar o seu consumo energ??tico e maximizar as suas poupan??as.
          """)



# In[74]:


st.subheader("Detalhes econ??micos")
budget = st.number_input("Defina aqui o seu or??amento m??ximo para reabilita????o", min_value=0, value=2000)
imi = st.number_input("Actualmente, quanto paga de IMI?", value=300)
private_imi = st.checkbox("Se n??o quiser providenciar esta informa????o, a ferramenta pode estimar um valor com base nos detalhes preenchidos.")
st.write("---")


# In[75]:


col41, col42, col43 = st.columns(3)
start_opt = col42.button("Clique aqui para come??ar")
# if start_opt:
#     with st.spinner("""A calcular o seu certificado energ??tico..."""):
#         et_r.fit(X_train, y_train["R"])
#         preds_r = et_r.predict(X_test)
#         mse_r = mean_squared_error(y_test["R"], preds_r)

#         et_ntc.fit(X_train, y_train["Ntc Valor"])
#         preds_ntc = et_ntc.predict(X_test)
#         mse_ntc = mean_squared_error(y_test["Ntc Valor"], preds_ntc)

#         et_nic.fit(X_train, y_train["Nic Valor"])
#         preds_nic = et_nic.predict(X_test)
#         mse_nic = mean_squared_error(y_test["Nic Valor"], preds_nic)

#         et_nvc.fit(X_train, y_train["Nvc Valor"])
#         preds_nvc = et_nvc.predict(X_test)
#         mse_nvc = mean_squared_error(y_test["Nvc Valor"], preds_nvc)
    # else:
    #     mse_r = 0
    #     mse_ntc = 0
    #     mse_nic = 0
    #     mse_nvc = 0
    #     st.write("")


# In[76]:


#full_user_data


# In[77]:


#model_inputs


# In[78]:


from platypus import *
#we have [0, 1] wall retrofit, [0,1] floor, [0,1] roof, window, AQS, ac
#problem_types = [walls, floors, roofs, windows, aqs, ac]


# In[79]:


if start_opt:
    def problem_types_init():
        problem_types = {"Paredes":-1, "Cobertura":-1, "Pavimento":-1, "Envidra??ados":-1, "AQS":-1, "AQS fonte de energia":-1, "AC":-1, "AC fonte de energia":-1}

        #wall variables
        if "com isolamento termico pelo exterior" in str(full_user_data["wall_type"].iloc[0]):
            problem_types["Paredes"] = -1
        else:
            problem_types["Paredes"] = Integer(0, 1)

        #roof variables
        if "-1" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Cobertura"] = -1
        elif "com isolamento" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Cobertura"] = -1
        else:
            problem_types["Cobertura"] = Integer(0, 2)

        #floor variables
        if "-1" in str(full_user_data["floor_type"].iloc[0]):
            problem_types["Pavimento"] = -1
        elif "com isolamento" in str(full_user_data["floor_type"].iloc[0]):
            problem_types["Pavimento"] = -1
        else:
            problem_types["Pavimento"] = Integer(0, 1)

        #window variables
        problem_types["Envidra??ados"] = Integer(0, 2)
        # AQS types variables
        problem_types["AQS"] = Integer(0, 5)
        # AQS source variables
        problem_types["AQS fonte de energia"] = Integer(0,1)
        #AC types variables
        problem_types["AC"] = Integer(0, 5)
        #AC source variables
        problem_types["AC fonte de energia"] = Integer(0,1)
        return problem_types
    # else:
    #     mse_r = 0
    #     mse_ntc = 0
    #     mse_nic = 0
    #     mse_nvc = 0
    #     st.write("")


# In[80]:


if start_opt:
   problem_types_d = problem_types_init()
   problem_types_label = []
   problem_types = []
   for i in problem_types_d:
       if problem_types_d[i] != -1:
           problem_types = np.append(problem_types, problem_types_d[i])
           problem_types_label = np.append(problem_types_label, i)
   # else:
   #     mse_r = 0
   #     mse_ntc = 0
   #     mse_nic = 0
   #     mse_nvc = 0
   #     st.write("")


# problem_types_label

# In[81]:


def r_to_levels(r_old, r_new): #This function tests wether or not a retrofit improved two or more levels
    if r_old < r_new:
        return False
    elif r_old <= 0.75 and r_old > 0.5 and r_new <= 0.25: #B para A+
        return True
    elif r_old <= 1 and r_old > 0.75 and r_new <= 0.5: #B- para A
        return True
    elif r_old <= 1.5 and r_old > 1 and r_new <= 0.75: #C para B
        return True
    elif r_old <= 2 and r_old > 1.5 and r_new <= 1: #D para B-
        return True
    elif r_old <= 2.5 and r_old > 2 and r_new <= 1.5: #E para C
        return True
    elif r_old <= 3 and r_old > 2.5 and r_new <= 2: #F para D
        return True
    else:
        return False


# In[82]:


def retrofits(df, x, problem_types_label):
    cost = [0]
    cost_wgov = [0]
    for value, label in zip(x, problem_types_label):
        #wall retrofits
        if label == "Paredes":
            if value == 1:
                wall_cost = 41*df["walls_area"].iloc[0]
                lim = 4500
                gov_ratio = 0.65
                df["walls_type"] = 0
                df["walls_u"] = 0.455
                if wall_cost*gov_ratio <= lim:
                    cost = np.append(cost, wall_cost*(1-gov_ratio))
                    cost_wgov = np.append(cost_wgov, wall_cost)
                elif wall_cost*gov_ratio > lim:
                    cost = np.append(cost, wall_cost-lim)
                    cost_wgov = np.append(cost_wgov, wall_cost)
                
        #roof retrofits        
        elif label == "Cobertura":
            roof_eps_cost = 13.5*df["roofs_area"].iloc[0]
            roof_xps_cost = 25*df["roofs_area"].iloc[0]
            lim = 4500
            gov_ratio = 0.65
            if value == 1: #EPS
                if "Cobertura inclinada" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 3 
                    df["roofs_u"] = 0.365
                elif "Cobertura horizontal" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 0
                    df["roofs_u"] = 0.365
                if roof_eps_cost*gov_ratio <= lim:
                    cost = np.append(cost, roof_eps_cost*(1-gov_ratio))
                elif roof_eps_cost*gov_ratio > lim:
                    cost = np.append(cost, roof_eps_cost-lim)

            elif value == 2: #XPS
                if "Cobertura inclinada" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 4 
                    df["roofs_u"] = 0.326
                elif "Cobertura horizontal" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 1
                    df["roofs_u"] = 0.326
                if roof_xps_cost*gov_ratio <= lim:
                    cost = np.append(cost, roof_xps_cost*(1-gov_ratio))
                elif roof_xps_cost*gov_ratio > lim:
                    cost = np.append(roof_xps_cost-lim)
                
        #Floors retrofits
        elif label == "Pavimento":
            if value == 1:
                floors_eps_cost = 13.5*df["floors_area"].iloc[0]
                lim = 4500
                gov_ratio = 0.65
                if floors_eps_cost*gov_ratio <= lim:
                    cost = np.append(cost, floors_eps_cost*(1-gov_ratio))
                elif floors_eps_cost*gov_ratio > lim:
                    cost = np.append(floors_eps_cost-lim)
                if "Pavimento em contacto com o solo" in str(full_user_data["floor_type"].iloc[0]):
                    df["floors_type"] = 2
                    df["floors_u"] = 0.32
                elif "Pavimento interior" in str(full_user_data["floor_type"].iloc[0]):
                    df["floors_type"] = 4
                    df["floors_u"] = 0.32
                else:
                    df["floors_type"] = 0
                    df["floors_u"] = 0.32
                    
        #windows retrofits
        elif label == "Envidra??ados":
            window_alu_cost = 380*df["window_area"].iloc[0]
            window_pvc_cost = 260*df["window_area"].iloc[0]
            gov_ratio = 0.85
            lim = 1500
            if value == 1: #Corte t??rmico alum??nio, duplo
                df["window_type"] = 3
                df["window_u"] = 2.5
                if window_alu_cost*gov_ratio <= lim:
                    cost = np.append(cost, window_alu_cost*(1-gov_ratio))
                elif window_alu_cost*gov_ratio > lim:
                    cost = np.append(cost, window_alu_cost-lim)
            elif value == 2: #PVC, DUplo
                df["window_type"] = 9
                df["window_u"] = 2.7
                if window_pvc_cost*gov_ratio <= lim:
                    cost = np.append(cost, window_pvc_cost*(1-gov_ratio))
                elif window_pvc_cost*gov_ratio > lim:
                    cost = np.append(cost, window_pvc_cost-lim)
                    
        #AQS types retrofits
        
        elif label == "AQS":

            if value == 1: #Esquentador
                if df["dhw_equipment"].iloc[0] != 2:
                    esq_cost = 450 #per unit
                    df["dhw_equipment"] = 2
                    df["nr_dhw_units"] = 1
                    cost = np.append(cost, esq_cost)
            elif value == 2: #Termoacumulador
                if df["dhw_equipment"].iloc[0] != 10:
                    ter_cost = 175
                    df["dhw_equipment"] = 10
                    df["nr_dhw_units"] = 1
                    cost = np.append(cost, ter_cost)
            elif value == 3: #Caldeira
                if df["dhw_equipment"].iloc[0] != 0:
                    cal_cost = 1750
                    df["dhw_equipment"] = 0
                    df["nr_dhw_units"] = 1
                    cost = np.append(cost, cal_cost)
            elif value == 4: #Bomba de calor - multi split-
                if df["dhw_equipment"].iloc[0] != 3:
                    bom_cost = 3750
                    gov_ratio = 0.85
                    lim=2500
                    df["dhw_equipment"] = 3
                    df["nr_dhw_units"] = 1
                cost = np.append(cost, bom_cost-lim)
            elif value == 5:#Painel Solar
                if df["dhw_equipment"].iloc[0] != 5:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["dhw_equipment"] = 5
                    if df["Tipologia"].iloc[0] <= 3:
                        df["nr_dhw_units"] = 3
                        cost = np.append(cost, sol_cost_3-lim)
                    elif df["Tipologia"].iloc[0] > 3:
                        df["nr_dhw_units"] = 6
                        cost = np.append(cost, sol_cost_6-lim)

        #AC types
        
                    
        elif label == "AC":
            if df["dhw_equipment"].iloc[0] == 3:
                df["ac_equipment"] = 5
                df["nr_ac_units"] = df["Tipologia"]
                cost = np.append(cost, 300*df["Tipologia"])          
            elif value == 1: #Esquentador
                cost = np.append(cost, 300*df["Tipologia"])
                if df["ac_equipment"].iloc[0] != 4:
                    esq_cost = 450 #per unit
                    df["ac_equipment"] = 4
                    df["nr_ac_units"] = 1
                    cost = np.append(cost, esq_cost)
                
            elif value == 2: #Termoacumulador
                if df["ac_equipment"].iloc[0] != 14:
                    ter_cost = 175
                    df["ac_equipment"] = 14
                    df["nr_ac_units"] = 1
                    cost = np.append(cost, ter_cost)
    
            elif value == 3:#Caldeira Mural + Radiadores fixos
                if df["ac_equipment"].iloc[0] != 8:
                    cal_cost = 2250
                    rad_cost = 15
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_equipment"] = 8
                    if df["Tipologia"].iloc[0] == 0:
                        df["nr_ac_units"] = 1
                    elif df["Tipologia"].iloc[0] > 0:
                        df["nr_ac_units"] = df["Tipologia"]
                
                    if df["dhw_equipment"].iloc[0] == 0:
                        if rad_cost*df["nr_ac_units"].iloc[0]*gov_ratio <= lim:
                            cost = np.append(cost, rad_cost*df["nr_ac_units"].iloc[0]*(1-gov_ratio))
                        elif (rad_cost*df["nr_ac_units"].iloc[0])*gov_ratio > lim:
                            cost = np.append(cost, rad_cost*df["nr_ac_units"].iloc[0]-2500)

                    elif df["dhw_equipment"].iloc[0] != 0:
                        if (cal_cost+rad_cost*df["nr_ac_units"].iloc[0])*gov_ratio <= lim:
                            cost = np.append(cost, cal_cost+(rad_cost*df["nr_ac_units"].iloc[0])*(1-gov_ratio))
                        elif (cal_cost+rad_cost*df["nr_ac_units"].iloc[0])*gov_ratio > lim:
                            cost = np.append(cost, cal_cost+(rad_cost*df["nr_ac_units"].iloc[0])-2500)
                            
            elif value == 4:#Multi-split
                if df["ac_equipment"].iloc[0] != 5:
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_equipment"] = 5
                    df["nr_ac_units"] = df["Tipologia"].iloc[0]
                    if 300*df["Tipologia"].iloc[0]*0.85 <= lim:
                        cost = np.append(cost, 300*df["Tipologia"].iloc[0]*gov_ratio)
                    elif 300*df["Tipologia"]*0.85 > lim:
                        cost = np.append(cost, 300*df["Tipologia"].iloc[0]-300*df["Tipologia"].iloc[0]*gov_ratio)
                        
            elif value == 5:#Painel Solar
                if df["ac_equipment"].iloc[0] != 7:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_equipment"] = 7
                    if df["Tipologia"].iloc[0] <= 3:
                        df["nr_ac_units"] = 3
                        if df["dhw_equipment"].iloc[0] != 5:
                            cost = np.append(cost, sol_cost_3-lim)
                    elif df["Tipologia"].iloc[0] > 3:
                        df["nr_ac_units"] = 6
                        if df["dhw_equipment"].iloc[0] != 5:
                            cost = np.append(cost, sol_cost_6-lim)

                
        #AQS source
        
        elif label == "AQS fonte de energia":
            if value == 1:
                if df["dhw_source"].iloc[0] != 10:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["dhw_source"] = 10
                    if df["dhw_equipment"].iloc[0] != 5 and df["ac_equipment"].iloc[0] != 7:
                        if df["Tipologia"].iloc[0] <= 3:
                            cost = np.append(cost, sol_cost_3-lim)
                        elif df["Tipologia"].iloc[0] > 3:
                            cost = np.append(cost, sol_cost_6-lim)
                
        #AC source
        
        elif label == "AC fonte de energia":
            if value == 1:
                if df["ac_source"].iloc[0] != 11:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_source"] = 11
                    if df["dhw_equipment"].iloc[0] != 5 and df["ac_equipment"].iloc[0] != 7 and df["dhw_source"].iloc[0] != 10:
                        if df["Tipologia"].iloc[0] <= 3:
                            cost = np.append(cost, sol_cost_3-lim)
                        elif df["Tipologia"].iloc[0] > 3:
                            cost = np.append(cost, sol_cost_6-lim)
    df["cost"] = np.sum(cost).ravel()
    return df


# In[189]:


def epc_opt(x):
    epc = model_inputs
    new_epc = epc.copy()
    final_epc = retrofits(new_epc, x, problem_types_label)
    area_calc = final_epc["??rea ??til de Pavimento"].iloc[0]
    cost = final_epc["cost"].iloc[0]
    
    original_r = et_r.predict(epc)[0]
    original_ntc = et_ntc.predict(epc)[0]   


    new_r = et_r.predict(final_epc.drop("cost", axis=1))[0]
    new_ntc = et_ntc.predict(final_epc.drop("cost", axis=1))[0]  

    energy_savings = original_ntc - new_ntc  #kWh/m2
    full_savings = energy_savings*area_calc #kWh
    savings = full_savings*0.22
    if r_to_levels(original_r, new_r):
        final_savings = savings + imi*0.25
    else:
        final_savings = savings
        
    cf1 = -cost + final_savings
    cf2 = cf1 + final_savings
    cf3 = cf2 + final_savings
    roi = cf3/(cost+0.000001)
#     df = pd.DataFrame(np.hstack((x, r_to_levels(original_r, new_r), original_ntc, new_ntc, final_savings, cost))).T
#     df.columns = np.hstack((problem_types_label, "IMI Bonus", "original Ntc", "new_ntc", "savings", "cost"))
#     df.to_csv(fname, mode='a', index=False, header=False)
    return [round(new_ntc*area_calc), round(-roi, 2), round(cost)]


# In[190]:


def epc_r(x):
    epc = model_inputs
    new_epc = epc.copy()
    final_epc = retrofits(new_epc, x, problem_types_label)
    area_calc = final_epc["??rea ??til de Pavimento"].iloc[0]
    cost = final_epc["cost"].iloc[0]
    
    original_r = et_r.predict(epc)[0]
    original_ntc = et_ntc.predict(epc)[0]   


    new_r = et_r.predict(final_epc.drop("cost", axis=1))[0]
    new_ntc = et_ntc.predict(final_epc.drop("cost", axis=1))[0]  

    energy_savings = original_ntc - new_ntc  #kWh/m2
    full_savings = energy_savings*area_calc #kWh
    savings = full_savings*0.22
    if r_to_levels(original_r, new_r):
        final_savings = savings + imi*0.25
    else:
        final_savings = savings
    cf1 = -cost + final_savings
    cf2 = cf1 + final_savings
    cf3 = cf2 + final_savings
    roi = cf3/(cost+0.000001)
#     df = pd.DataFrame(np.hstack((x, r_to_levels(original_r, new_r), original_ntc, new_ntc, final_savings, cost))).T
#     df.columns = np.hstack((problem_types_label, "IMI Bonus", "original Ntc", "new_ntc", "savings", "cost"))
#     df.to_csv(fname, mode='a', index=False, header=False)
    return new_r


# In[154]:


if start_opt:
    with st.spinner("""A realizar a optimiza????o de reabilita????o do seu Im??vel/Edif??cio..."""):
        problem = Problem(len(problem_types_label), 3)
        problem.types[:] = problem_types
        problem.function = epc_opt
        # fname =  "NSGAII" + time_str() + ".csv"
        # with open(fname, 'a'):
        algorithm = NSGAII(problem, population_size=25)
        algorithm.run(250)


# In[155]:


if start_opt:
    x = [s.objectives[0] for s in algorithm.result]
    y = [s.objectives[1] for s in algorithm.result]
    z = [s.objectives[2] for s in algorithm.result]
    results_df = pd.DataFrame([x, y, z]).transpose()
    results_df.columns = ["Ntc [kWh]", "ROI in 3 years [ratio]", "Retrofit cost [???]"]
    results_df["ROI in 3 years [ratio]"] = results_df["ROI in 3 years [ratio]"].apply(lambda x: -x)
    


# In[156]:


def r_to_epc(r):
    if r <= 0.25:
        return "A+"
    elif r <= 0.50:
        return "A"
    elif r <= 0.75:
        return "B"
    elif r <= 1.00:
        return "B-"
    elif r <= 1.50:
        return "C"
    elif r <= 2.00:
        return "D"
    elif r <= 2.50:
        return "E"
    else:
        return "F"


# In[159]:


if start_opt:
    sol = []
    for s in algorithm.result:
        for v in enumerate(problem_types):
            int_i = s.variables[v[0]]
            sol = np.hstack((sol, v[1].decode(int_i)))
        vars_opt = pd.DataFrame(sol.reshape(-1, len(problem_types)))
        vars_opt.columns = problem_types_label
        
        full_opt_df = pd.concat([vars_opt, results_df], axis=1).drop_duplicates()
        budget_max_df = full_opt_df[(full_opt_df["Retrofit cost [???]"] <= budget) & 
                            (full_opt_df["ROI in 3 years [ratio]"] > 0)].drop_duplicates(full_opt_df.columns[: len(full_opt_df.columns)-3])
        new_rs = []
        for i, j in enumerate(budget_max_df.index):
            new_rs = np.append(new_rs, epc_r(budget_max_df.drop(["Ntc [kWh]", "ROI in 3 years [ratio]", "Retrofit cost [???]"], axis=1).iloc[i]))
        budget_max_df["New R ratios"] = new_rs
        budget_max_df["New R ratios"] = budget_max_df["New R ratios"].apply(lambda x: r_to_epc(x))
        


# In[161]:


def retrofit_translate(df1):
    #walls
    df = df1.copy()
    for i in df.reset_index().index:
        for label in problem_types_label:
            if label == "Paredes":
                if df["Paredes"].iloc[i] == 1:
                    df["Paredes"].iloc[i] = "ETICS"
                else:
                    df["Paredes"].iloc[i]  = "-"
                    
            #roofs    
            elif label == "Cobertura":
                if df["Cobertura"].iloc[i]  == 1:
                    df["Cobertura"].iloc[i]  = "EPS"
                elif df["Cobertura"].iloc[i]  == 2:
                    df["Cobertura"].iloc[i]  = "XPS"
                else:
                    df["Cobertura"].iloc[i]  = "-"
    
            #floors
            elif label == "Pavimento":
                if df["Pavimento"].iloc[i]  == 1:
                    df["Pavimento"].iloc[i]  = "EPS"
                else:
                    df["Pavimento"].iloc[i]  = "-"
                    
            #windows    
            elif label == "Envidra??ados":
                if df["Envidra??ados"].iloc[i]  == 1:
                    df["Envidra??ados"].iloc[i]  = "PVC/vidro duplo"
                elif df["Envidra??ados"].iloc[i]  == 2:
                    df["Envidra??ados"].iloc[i]  = "Alum??nio c/ corte t??rmico/vidro duplo"
                else:
                    df["Envidra??ados"].iloc[i]  = "-"
            
            #aqs_t
            elif label == "AQS":
                if df["AQS"].iloc[i] == 1:
                    df["AQS"].iloc[i] = "Esquentador"
                elif df["AQS"].iloc[i] == 2:
                    df["AQS"].iloc[i] = "Termoacumulador"
                elif df["AQS"].iloc[i] == 3:
                    df["AQS"].iloc[i] = "Caldeira convencional"
                elif df["AQS"].iloc[i] == 4:
                    df["AQS"].iloc[i] = "Bomba de calor ar_??gua"
                elif df["AQS"].iloc[i] == 5:
                    df["AQS"].iloc[i] = "Pain??is solares"
                else:
                    df["AQS"].iloc[i] = "-"
            
            #ac_t
            elif label == "AC":
                if df["AQS"].iloc[i] == "Bomba de calor ar_??gua":
                    df["AC"].iloc[i] = "Bomba de calor ar_??gua"
                else:
                    if df["AC"].iloc[i] == 1:
                        df["AC"].iloc[i] = "Esquentador c/ Radiadores fixos"
                    elif df["AC"].iloc[i] == 2:
                        df["AC"].iloc[i] = "Termoacumulador c/ Radiadores fixos"
                    elif df["AC"].iloc[i] == 3:
                        df["AC"].iloc[i] = "Caldeira Mural c/ Radiadores Fixos"
                    elif df["AC"].iloc[i] == 4:
                        df["AC"].iloc[i] = "Multi-split"
                    elif df["AC"].iloc[i] == 5:
                        df["AC"].iloc[i] = "Pain??is solares"
                    else:
                        df["AC"].iloc[i] = "-"
            
            #aqs_s
            elif label == "AQS fonte de energia":
                if df["AQS fonte de energia"].iloc[i] == 1:
                    df["AQS fonte de energia"].iloc[i] = "Solar"
                else:
                    df["AQS fonte de energia"].iloc[i] = "-"
                    
            
            #ac_s
            elif label == "AC fonte de energia":
                if df["AC fonte de energia"].iloc[i] == 1:
                    df["AC fonte de energia"].iloc[i] = "Solar"
                else:
                    df["AC fonte de energia"].iloc[i] = "-"

    return df


# In[162]:


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# In[163]:


# import plotly.express as px
# if start_opt:
#     fig = px.scatter_3d(results_df[results_df["ROI in 3 years [ratio]"] >= 0], 
#                         x="Ntc [kWh]", 
#                         y="ROI in 3 years [ratio]", 
#                         z="Retrofit cost [???]",)
#                         #color="New R ratios",
#                         #color_continuous_scale=px.colors.diverging.Tealrose)

#     fig.update_layout(
#                         autosize=True,
#                         width=1200,
#                         height=800)
#     camera = dict(
#         up=dict(x=0, y=0, z=1.25),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=1.25, y=-1.25, z=1)
#     )

#     fig.update_layout(scene_camera=camera)
#     st.plotly_chart(fig)
#     fig.show()


# In[ ]:





# In[164]:


if start_opt:
    csv_df = budget_max_df.copy()
#     col51, col52, col53 = st.columns(3)
    csv_df.index = ["optimum solution " + str(i+1) for i, j in enumerate(csv_df.index)]
    string_table = retrofit_translate(csv_df)
    categories_pl = retrofit_translate(csv_df)
    csv = convert_df(string_table)
    categories_pl
    


# In[166]:


if start_opt:
    chart = categories_pl[["Ntc [kWh]", "ROI in 3 years [ratio]", "Retrofit cost [???]"]].reset_index(drop=True)
    bar_idx = []
    for i, j in enumerate(chart.index):

        bar_idx = np.append(bar_idx, "optimum solution " + str(i + 1))
    chart["solution"] = bar_idx
    bar_chart = chart.melt(id_vars="solution")


# In[ ]:





# In[167]:


if start_opt:
    import plotly.express as px
    fig = px.bar(bar_chart[bar_chart["variable"] == "Ntc [kWh]"], x="variable", y="value", barmode="group", color="solution", color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    #fig.show()


# In[168]:


if start_opt:
    import plotly.express as px
    fig = px.bar(bar_chart[bar_chart["variable"] == "ROI in 3 years [ratio]"], x="variable", y="value", barmode="group", color="solution", color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    #fig.show()


# In[169]:


if start_opt:
    import plotly.express as px
    fig = px.bar(bar_chart[bar_chart["variable"] == "Retrofit cost [???]"], x="variable", y="value", barmode="group", color="solution", color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig)
    #fig.show()


# In[ ]:




