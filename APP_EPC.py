#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required packages from python library
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

#st.set_page_config(layout="wide")


# In[2]:


pd.set_option("display.max_columns", None)


# # Model

# In[3]:


data = pd.read_csv("data.csv").drop("Unnamed: 0", axis=1)


# In[4]:


X = data.drop(["R", "Ntc Valor", "Nic Valor", "Nvc Valor", "EPC", "TARGET", "Ntc Limite"], axis=1)
y = data[["EPC", "R", "Ntc Valor", "Nic Valor", "Nvc Valor", "Ntc Limite"]]


# In[5]:


#X.columns


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[7]:


# et_r.fit(X_train, y_train["R"])
# preds = et_r.predict(X_test)
# r2_score(y_test["R"], preds)


# In[8]:



# et_ntc.fit(X_train, y_train["Ntc Valor"])
# preds = et_ntc.predict(X_test)
# r2_score(y_test["Ntc Valor"], preds)


# In[9]:


# et_ntcl.fit(X_train, y_train["Ntc Limite"])
# preds = et_ntcl.predict(X_test)
# r2_score(y_test["Ntc Limite"], preds)


# In[10]:


# et_nic.fit(X_train, y_train["Nic Valor"])
# preds = et_nic.predict(X_test)
# r2_score(y_test["Nic Valor"], preds)


# In[11]:


# et_nvc.fit(X_train, y_train["Nvc Valor"])
# preds = et_nvc.predict(X_test)
# r2_score(y_test["Nvc Valor"], preds)


# # Data

# In[12]:


def period_to_epoch(x):
    if pd.isna(x) == True:
        return "is null"
    if x == "before 1918":
        return 0
    if x == "between 1919 and 1945":
        return 1
    if x== "between 1946 andnd 1960":
        return 2
    if x== "between 1961 and 1970":
        return 3
    if x== "between 1971 and 1980":
        return 4
    if x== "between 1981 and 1990":
        return 5
    if x== "between 1991 and 1995":
        return 6
    if x== "between 1996 and 2000":
        return 7
    if x== "between 2001 and 2005":
        return 8
    else:
        return 9


# In[13]:


def epochs_to_period(x):
    if x == 0:
        return "before 1918"
    if x == 1:
        return "between 1919 and 1945"
    if x== 2:
        return "between 1946 and 1960"
    if x== 3:
        return "between 1961 and 1970"
    if x== 4:
        return "between 1971 and 1980"
    if x== 5:
        return "between 1981 and 1990"
    if x== 6:
        return "between 1991 and 1995"
    if x== 7:
        return "between 1996 and 2000"
    if x== 8:
        return "between 2001 and 2005"
    else:
        return "Posterior and 2005"


# In[14]:


period_df = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
period_df["label"] = period_df[0].apply(epochs_to_period)


# In[15]:


#period_df


# In[16]:


typology_type = ['> T6', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']


# In[17]:


typology_labels = [0, 1, 2, 3, 4, 5, 6, 7]


# In[18]:


typology_df = pd.DataFrame([typology_type, typology_labels]).T.apply(np.roll, shift=-1)


# In[19]:


epc_type = ['Building', 'Fraction (without horizontal property)', 'Fraction (horizontal property)']
epc_type_labels = [0,1, 2]
epc_type_df = pd.DataFrame([epc_type, epc_type_labels]).T


# In[20]:


district_types = pd.read_csv("disctrict_types.csv")


# In[21]:


wall_types = pd.read_csv("wall_types.csv")
roof_types = pd.read_csv("roof_types.csv")
floor_types = pd.read_csv("floors_types.csv")
window_types = pd.read_csv("window_types.csv")


# In[22]:


ac_sources = pd.read_csv("ac_sources.csv").iloc[:12]
ac_types = pd.read_csv("ac_types.csv").iloc[:16]

dhw_sources = pd.read_csv("dhw_sources.csv")
dhw_types = pd.read_csv("dhw_types.csv")


# ## Walls

# In[23]:


#wall_types


# In[45]:


epoch_walls = data.drop('EPC', axis=1).groupby("epoch").mean()["walls_type"].astype("int")


# In[46]:


#epoch_walls


# In[47]:


def period_to_wall(x):
    for wall in enumerate(epoch_walls):
        if period_to_epoch(x) == wall[0]:
            return wall[1]
            


# In[48]:


#period_to_wall("entre 2001 a 2005")


# ## ROOFS

# In[49]:



#roof_types


# In[50]:


epoch_roofs = data.drop('EPC', axis=1).groupby("epoch").mean()["roofs_type"].astype("int")


# In[51]:


#epoch_roofs


# In[52]:


def period_to_roof(x):
    for roof in enumerate(epoch_roofs):
        if period_to_epoch(x) == roof[0]:
            return roof[1]


# In[53]:


#period_to_roof("entre 2001 a 2005")


# ## Floors

# In[54]:



#floor_types["solution"]


# In[56]:


epoch_floors = data.drop('EPC', axis=1).groupby("epoch").mean()["floors_type"].astype("int")


# In[57]:


#epoch_floors


# In[58]:


def period_to_floor(x):
    for floor in enumerate(epoch_floors):
        if period_to_epoch(x) == floor[0]:
            return floor[1]


# In[59]:


#period_to_floor("entre 2001 a 2005")


# ## Windows

# In[60]:


#window_types


# In[61]:


epoch_windows = data.drop('EPC', axis=1).groupby("epoch").mean()["window_type"].astype("int")


# In[62]:


#epoch_windows


# In[63]:


def period_to_window(x):
    for window in enumerate(epoch_windows):
        if period_to_epoch(x) == window[0]:
            return  window[1]


# In[64]:


#period_to_window("entre 2001 a 2005")


# In[65]:


#ac_sources["0"]


# In[66]:


#ac_types["0"]


# In[67]:


#dhw_types


# In[68]:


#dhw_sources["0"]


# # Interface

# In[69]:


st.write("""
# Building and Home retrofitting assistant for the Portuguese building stock

This web application predicts a building or home energy performance certificate, energy indicators, and suggests the best retrofits within a specified budget limit.
""")
st.write("---")


# In[70]:


#ADJUST COLUMN WIDTH

# st.markdown(
# """
# <style>
# [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
# width: 600px;
# }
# [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
# width: 600px;
# margin-left: -600px;
# }
# </style>
# """,
# unsafe_allow_html=True
# )


# In[71]:


# Sidebar
# Header of Specify Input Parameters
st.header('Please specify the details of your home or building')

def user_base_input():
    st.subheader("General details")
    
    District = st.selectbox("Location", district_types["1"], index=6)
    B_type = st.selectbox("Type of certificate", 
                                  ["Building", "Horizontal property"], 
                                  index=1)
    if B_type == "Horizontal property":
        floor_position = st.selectbox("Floor location of your house", 
                                              ["Ground", "Middle", "Last"], 
                                              index=2)
        N_floors = st.number_input("Total number of floors in your building",
                                           step=1, 
                                           value=2)
    else:
        N_floors = st.number_input('Number of floors in your building', 
                                           step=1, value=5)
        floor_position = 0

    Period = st.selectbox('Construction period', period_df["label"], index=7)
    f_area = st.number_input('Area',value=100,  step=1)
    f_height = st.number_input('Floor height', value=2.80)
    typology = st.selectbox('Typology', typology_df[0], index=3)

    st.subheader("Climatization and thermal comfort details")

    ac_type = st.selectbox("Type of climatization equipment", 
                           ac_types["0"].append(pd.Series("Do not have any"), 
                                                ignore_index=True), 
                           index=10) #16 - nao possuo
    
    if ac_type != "Do not have any":
        ac_source = st.selectbox("Type of energy source for climatization", ac_sources["0"], 
                                         index=4)
        nr_ac_units = st.number_input("Number of HVAC equipments", 
                                      value=2, 
                                      step=1,
                                      min_value=1)
    else:
        ac_source = -1
        nr_ac_units = -1

    st.subheader("Domestic Hot Water details")

    dhw_type = st.selectbox('Type of DHW equipment', 
                            dhw_types["0"].append(pd.Series("Do not have any"), 
                                                  ignore_index=True), 
                            index= 8)
    if dhw_type != "Do not have any":
        dhw_source = st.selectbox("Type of energy source for DHW", dhw_sources["0"], 
                                         index=4)
        nr_dhw_units = st.number_input("Number of DHW equipments", 
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


# In[72]:



base_inputs = user_base_input()
#base_inputs


# In[73]:


# st.sidebar.write("---")
# st.sidebar.caption("Para proceder com a previsão do seu certificado energético precisamos de dados um pouco mais detalhados. No entanto, podemos simplificar estes dados com base no ano de construção do seu imóvel ou edifício, e as soluções construtivas típicas para essa época de construção.")
# st.sidebar.caption("Este processo pode gerar informações incorrectas particularmente se o seu imóvel ou edíficio já sofreu obras de reabilitação, aumentando assim o erro médio da previsão.")


# In[74]:


st.write("---")
st.caption("To proceed with the prediction of your energy performance certificate we need a little more detailed data. However, we can simplify this data based on the year of construction of your property or building, and the typical construction solutions for that period.")
st.caption("This process can generate incorrect information particularly if your property or building has already undergone rehabilitation works, thus increasing the average error of the forecast.")


# In[132]:


# df1 = proceeder()
# df1


# In[133]:


#floor_types


# In[134]:


def user_advanced_inputs():
    with st.expander("Advanced Details"):
        base_wall_type = period_to_wall(base_inputs["Period"].iloc[0])
        base_roof_type = period_to_roof(base_inputs["Period"].iloc[0])
        base_floor_type = period_to_floor(base_inputs["Period"].iloc[0])
        base_window_type = period_to_window(base_inputs["Period"].iloc[0])
        base_wall_area = np.sqrt(base_inputs["f_area"].iloc[0])*base_inputs["f_height"].iloc[0]*2
        
        st.subheader("Wall details")
        #WALLS        
        wall_type = st.selectbox("Wall construction type:", 
                                 wall_types["Solution"], 
                                 index= base_wall_type)
        wall_area = st.number_input("Exterior wall area", 
                                    value= base_wall_area)
        #ROOF
        st.subheader("Roof details")
        
        
        if base_inputs["B_type"].iloc[0] == "Building": #B_type
            roof_type = st.selectbox("Roof construction type:", 
                                     roof_types["Solution"],
                                     index= base_roof_type)
        elif base_inputs["floor_position"].iloc[0] == "Last": #floor_position
            roof_type = st.selectbox("Roof construction type:", 
                                     roof_types["Solution"],
                                     index=base_roof_type)
        else:
            roof_type = -1
            st.caption("Your certificate does not need this data to be predicted.")
            
        if roof_type != -1:
            roof_area = st.number_input("Roof area:", 
                                        value= base_inputs["f_area"].iloc[0])
        else:
            roof_area = -1
            
        st.subheader("Floor details")
        #FLOORS
        if base_inputs["B_type"].iloc[0] == "Building": #B_type
            floor_type = st.selectbox("Floor construction type:", 
                                      floor_types["solution"], 
                                      index=base_floor_type)
        elif base_inputs["floor_position"].iloc[0] == "Ground": #floor_position
            floor_type = st.selectbox("Floor construction type:", 
                                     floor_types["solution"], 
                                     index=base_floor_type)
        else:
            floor_type = -1
            st.caption("Your certificate does not need this data to be predicted.")
            
        if floor_type!= -1:
            floor_area = st.number_input("Floor area:", 
                                        value= base_inputs["f_area"].iloc[0])
            
        else:
            floor_area=-1
            

        st.subheader("Window details")
        window_area = st.number_input("Window area:", value=wall_area*0.2)
        window_type = st.selectbox("Window construction type:", 
                                   window_types["Tipo de Solução 1"],
                                   index=base_window_type)


    df2 = pd.DataFrame([wall_type, wall_area, roof_type, roof_area, floor_type, floor_area, window_type, window_area]).T #, ac_type, nr_ac_units, dhw_type, nr_dhw_units]).T
    df2.columns = ["wall_type", "wall_area", "roof_type", "roof_area", "floor_type", "floor_area", "window_type", "window_area"] #, "ac_type", "nr_ac_units", "dhw_type", "nr_dhw_units"]
    return df2


# In[135]:


advanced_inputs = user_advanced_inputs()
#advanced_inputs


# In[136]:


full_user_data = pd.concat([base_inputs, advanced_inputs],axis=1)


# In[137]:


#full_user_data


# In[138]:


def district_to_int(x):
    for i in enumerate(district_types["1"]):
        if x == i[1]:
            return i[0]

def tipo_int(x):
    if x== "Horizontal property":
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
            return wall_types["Coeficiente Solução"].iloc[i[0]]

def wall_type_to_int(x):
    for i in enumerate(wall_types["Solution"]):
        if x == i[1]:
            return i[0]

def roof_type_to_u(x):
    for i in enumerate(roof_types["Solution"]):
        if x ==-1:
            return -1
        elif x == i[1]:
            return roof_types["Coeficiente Solução"].iloc[i[0]]

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
            return floor_types["Coeficiente Solução"].iloc[i[0]]

def floor_type_to_int(x):
    for i in enumerate(floor_types["solution"]):
        if x ==-1:
            return -1
        elif x == i[1]:
            return i[0]

def window_type_to_u(x):
    for i in enumerate(window_types["Tipo de Solução 1"]):
        if x == i[1]:
            return window_types["Uwdn"].iloc[i[0]]

def window_type_to_int(x):
    for i in enumerate(window_types["Tipo de Solução 1"]):
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
        if x == "Do not have any":
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
        if x == "Do not have any":
            return -1
        elif x == i[1]:
            return i[0]


# In[139]:


model_inputs = pd.DataFrame(np.repeat(0, 25)).T
model_inputs.columns = X_train.columns


# In[140]:


model_inputs["Distrito"] = district_to_int(full_user_data["District"].iloc[0])
model_inputs["Tipo de Imóvel"] = tipo_int(full_user_data["B_type"].iloc[0])
model_inputs["Área útil de Pavimento"] = full_user_data["f_area"].iloc[0]
model_inputs["Pé Direito Médio"] = full_user_data["f_height"].iloc[0]
model_inputs["Tipologia"] = assoa_int(full_user_data["typology"].iloc[0])
model_inputs["Número Total de Pisos"] = full_user_data["N_floors"]
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


# In[141]:


#model_inputs


# In[142]:


# user_view_inputs = full_user_data.copy()
# user_view_inputs.columns =  ["Tipo", 
#                              "Posição do piso", 
#                              "Número de pisos",
#                              "Período de construção",
#                              "Área útil",
#                              "Pé direito",
#                              "Assoalhadas",
#                              "Tipo de Climatização",
#                              "Fonte energia Climatização",
#                              "Número de unidades para Aquecimento",
#                              "Tipo de AQS",
#                              "Fonte energia AQS",
#                              "Número de equipamentos para AQS",
#                              "Tipo de paredes",
#                              "Área das paredes",
#                              "Tipo de Cobertura",
#                              "Área de Cobertura",
#                              "Tipo de Pavimento",
#                              "Área de Pavimento",
#                              "Tipo de Envidraçados",
#                              "Área de Envidraçados"]

# st.write("Aqui pode consultar os dados usados para a previsão do seu certificado:")
# st.dataframe(user_view_inputs.T.astype("str"))


# # Model Generation

# In[143]:


r_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
ntc_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
nic_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
nvc_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)


# In[144]:


# with st.spinner("""O cálculo do seu certificado não substitui a avaliação realizada por um perito.
#                 As informações aqui avançadas representam uma aproximação ao cálculo do certificado energético 
#                 com um erro médio de uma classe energética."""):
# r_model.fit(X_train, y_train["R"])
# ntc_model.fit(X_train, y_train["Ntc Valor"])
# nic_model.fit(X_train, y_train["Nic Valor"])
# nvc_model.fit(X_train, y_train["Nvc Valor"])


# In[145]:


y


# # K-Fold Cross-Validation

# In[160]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Load the dataset (example using the iris dataset)


# Create an instance of the machine learning model

# Define the number of folds for cross-validation
num_folds = 6

# Create a k-fold cross-validation object
kfold = KFold(n_splits=num_folds)

# Initialize lists to store R2 and RMSE scores
r2_scores_r = []
rmse_scores_r = []
mape_scores_r = []
# Perform cross-validation
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y["R"].iloc[train_index], y["R"].iloc[test_index]
    
    # Fit the model to the training data
    r_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = r_model.predict(X_test)
    
    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores_r.append(r2)
    
    # Calculate RMSE score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores_r.append(rmse)
    
    # Calculate MAPE score
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_scores_r.append(mape)

# Print the R2 scores for each fold
for fold, score in enumerate(r2_scores_r):
    print(f"Fold {fold+1}: R2 = {score}")

# Print the RMSE scores for each fold
for fold, score in enumerate(rmse_scores_r):
    print(f"Fold {fold+1}: RMSE = {score}")
    
# Print the MAPE scores for each fold
for fold, score in enumerate(mape_scores_r):
    print(f"Fold {fold+1}: MAPE = {score}")

# Calculate and print the mean R2 score
mean_r2 = np.mean(r2_scores_r)
print(f"Mean R2: {mean_r2}")

# Calculate and print the mean RMSE score
mean_rmse = np.mean(rmse_scores_r)
print(f"Mean RMSE: {mean_rmse}")

# Calculate and print the mean MAPE score
mean_mape = np.mean(mape_scores_r)
print(f"Mean MAPE: {mean_mape}")


# In[153]:


# Initialize lists to store R2 and RMSE scores
r2_scores_ntc = []
rmse_scores_ntc = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y["Ntc Valor"].iloc[train_index], y["Ntc Valor"].iloc[test_index]
    
    # Fit the model to the training data
    ntc_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = ntc_model.predict(X_test)
    
    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores_ntc.append(r2)
    
    # Calculate RMSE score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores_ntc.append(rmse)

# Print the R2 scores for each fold
for fold, score in enumerate(r2_scores_ntc):
    print(f"Fold {fold+1}: R2 = {score}")

# Print the RMSE scores for each fold
for fold, score in enumerate(rmse_scores_ntc):
    print(f"Fold {fold+1}: RMSE = {score}")

# Calculate and print the mean R2 score
mean_r2 = np.mean(r2_scores_ntc)
print(f"Mean R2: {mean_r2}")

# Calculate and print the mean RMSE score
mean_rmse = np.mean(rmse_scores_ntc)
print(f"Mean RMSE: {mean_rmse}")


# In[154]:


# Initialize lists to store R2 and RMSE scores
r2_scores_nvc = []
rmse_scores_nvc = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y["Nvc Valor"].iloc[train_index], y["Nvc Valor"].iloc[test_index]
    
    # Fit the model to the training data
    nvc_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = nvc_model.predict(X_test)
    
    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores_nvc.append(r2)
    
    # Calculate RMSE score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores_nvc.append(rmse)

# Print the R2 scores for each fold
for fold, score in enumerate(r2_scores_nvc):
    print(f"Fold {fold+1}: R2 = {score}")

# Print the RMSE scores for each fold
for fold, score in enumerate(rmse_scores_nvc):
    print(f"Fold {fold+1}: RMSE = {score}")

# Calculate and print the mean R2 score
mean_r2 = np.mean(r2_scores_nvc)
print(f"Mean R2: {mean_r2}")

# Calculate and print the mean RMSE score
mean_rmse = np.mean(rmse_scores_nvc)
print(f"Mean RMSE: {mean_rmse}")


# In[155]:


# Initialize lists to store R2 and RMSE scores
r2_scores_nic = []
rmse_scores_nic = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y["Nic Valor"].iloc[train_index], y["Nic Valor"].iloc[test_index]
    
    # Fit the model to the training data
    nic_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = nic_model.predict(X_test)
    
    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores_nic.append(r2)
    
    # Calculate RMSE score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores_nic.append(rmse)

# Print the R2 scores for each fold
for fold, score in enumerate(r2_scores_nic):
    print(f"Fold {fold+1}: R2 = {score}")

# Print the RMSE scores for each fold
for fold, score in enumerate(rmse_scores_nic):
    print(f"Fold {fold+1}: RMSE = {score}")

# Calculate and print the mean R2 score
mean_r2 = np.mean(r2_scores_nic)
print(f"Mean R2: {mean_r2}")

# Calculate and print the mean RMSE score
mean_rmse = np.mean(rmse_scores_nic)
print(f"Mean RMSE: {mean_rmse}")


# In[79]:


pd.DataFrame([r2_scores_r, 
              rmse_scores_r, 
              r2_scores_ntc, 
              rmse_scores_ntc, 
              r2_scores_nic, 
              rmse_scores_nic, 
              r2_scores_nvc, 
              rmse_scores_nvc]).transpose().round(2)


# # END

# In[162]:


col_a, col_c, colb = st.columns(3)
simulate_button = col_c.button('Predict energy indicators!')
#if simulate_button:
with st.spinner("""The calculation of your certificate does not replace the assessment carried out by an expert.
                The information provided here represents an approximation to the calculation of the energy certificate 
                with an average error of an energy class and in no way binding to official results."""):
    @st.cache_resource()  # 👈 Added this
    def r_():
        return r_model.fit(X_train, y_train["R"])


    @st.cache_resource()  # 👈 Added this
    def ntc_():
        return ntc_model.fit(X_train, y_train["Ntc Valor"])

    @st.cache_resource()  # 👈 Added this
    def nvc_():
        return nvc_model.fit(X_train, y_train["Nvc Valor"])

    @st.cache_resource()  # 👈 Added this
    def nic_():
        return nic_model.fit(X_train, y_train["Nic Valor"])

et_r = r_() 
et_ntc =  ntc_()
et_nvc =  nvc_() 
et_nic =  nic_() 

# et_r = r_model
# et_ntc =  ntc_model
# et_nvc =  nic_model 
# et_nic =  nvc_model

# else:
#     mse_nvc = 0
#     mse_nic = 0
#     mse_ntc = 0
#     mse_r = 0
#     st.write("")
#     st.header('Carregue no botão da barra lateral para iniciar a previsão do seu Certificado Energético')


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

# In[163]:


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


# In[164]:


area_calc = model_inputs["Área útil de Pavimento"].iloc[0]


# In[165]:


if simulate_button:

    r = np.round(et_r.predict(model_inputs)[0], 2)
    ntc = np.round(et_ntc.predict(model_inputs)[0]*area_calc,0)
    nic = np.round(et_nic.predict(model_inputs)[0]*area_calc, 0)
    nvc = np.round(et_nvc.predict(model_inputs)[0]*area_calc,0)
    
    col_image1, col_image_c, col_image2 = st.columns(3)
    col_image_c.image(r_to_epc_fig(r))
    
    col1, col2, col3 = st.columns(3)

    
    col1.image("epcs/cooling.png", width=35)
    col1.metric("Cooling energy (kWh/year)", str(round(int(nvc/1000), 0)) + " k")
    col2.image("epcs/heating.png", width=35)
    col2.metric("Heating energy (kWh/year)", str(round(int(nic/1000), 0)) + " k")
    col3.image("epcs/en.png", width=35)
    col3.metric("Total energyl (kWh/year)", str(round(int(ntc/1000), 0)) + " k")
    st.metric("R ratio", r)


# In[ ]:





# # Optimization

# In[166]:


st.write("---")
st.write("""Energy certificates categorise buildings according to their energy efficiency,
             in order to identify and promote the rehabilitation of the most severe cases. As such, the government approved
             a package of measures aimed at encouraging such rehabilitation: the "Environmental Contribution Fund".
          """)

st.write("""
         The environmental contribution fund covers some expenses for the rehabilitation of buildings, according to
         a total set of measures defined by the government (e.g., the application of solar panels, 
         insulation of roofs and walls, change of glazing, among others).
          """)

st.write("""
         In addition, the government provides a discount or exemption of taxes through certain cases of
         energy certification that improve significantly after rehabilitation.
          """)

st.write("""
         In view of this information, if you want to carry out rehabilitation interventions in your property and / or building, 
         you can use this tool, which suggests optimal combinations of rehabilitations based on your maximum budget to minimise energy 
         consumption and maximise savings.
          """)



# In[167]:


st.subheader("Economic details")
budget = st.number_input("Here you can stipulate your maximum rehabilitation budget", min_value=0, value=4500)
imi = st.number_input("Presently, how mucxh do you pay for housing taxes?", value=300)
private_imi = st.checkbox("If you do not want to provide this information, the tool can estimate a value based on the information provided.")
st.write("---")


# In[168]:


col41, col42, col43 = st.columns(3)
start_opt = col42.button("Click here to start")
# if start_opt:
#     with st.spinner("""A calcular o seu certificado energético..."""):
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


# In[169]:


#full_user_data


# In[170]:


#model_inputs


# In[171]:


from platypus import *
#we have [0, 1] wall retrofit, [0,1] floor, [0,1] roof, window, AQS, ac
#problem_types = [walls, floors, roofs, windows, aqs, ac]


# In[172]:


if start_opt:
    def problem_types_init():
        problem_types = {"Walls":-1, "Roof":-1, "Floor":-1, "Glazing":-1, "DHW":-1, "DHW energy source":-1, "HVAC":-1, "HVAC energy source":-1}

        #wall variables
        if "with exterior insulation" in str(full_user_data["wall_type"].iloc[0]):
            problem_types["Walls"] = -1
        else:
            problem_types["Walls"] = Integer(0, 1)

        #roof variables
        if "-1" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Roof"] = -1
        elif "with exterior insulation" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Roof"] = -1
        elif "with interior insulation" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Roof"] = -1
        elif "with insulation" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Roof"] = -1
        elif "with slab insulation" in str(full_user_data["roof_type"].iloc[0]):
            problem_types["Roof"] = -1
        else:
            problem_types["Roof"] = Integer(0, 2)

        #floor variables
        if "-1" in str(full_user_data["floor_type"].iloc[0]):
            problem_types["Floor"] = -1
        elif "with insulation" in str(full_user_data["floor_type"].iloc[0]):
            problem_types["Floor"] = -1
        else:
            problem_types["Floor"] = Integer(0, 1)

        #window variables
        problem_types["Glazing"] = Integer(0, 2)
        # AQS types variables
        problem_types["DHW"] = Integer(0, 5)
        # AQS source variables
        problem_types["DHW energy source"] = Integer(0,1)
        #AC types variables
        problem_types["HVAC"] = Integer(0, 5)
        #AC source variables
        problem_types["HVAC energy source"] = Integer(0,1)
        return problem_types
    # else:
    #     mse_r = 0
    #     mse_ntc = 0
    #     mse_nic = 0
    #     mse_nvc = 0
    #     st.write("")


# In[173]:


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


# In[174]:


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


# In[175]:


def retrofits(df, x, problem_types_label):
    cost = [0]
    cost_wgov = [0]
    for value, label in zip(x, problem_types_label):
        #wall retrofits
        if label == "Walls":
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
        elif label == "Roof":
            roof_eps_cost = 13.5*df["roofs_area"].iloc[0]
            roof_xps_cost = 25*df["roofs_area"].iloc[0]
            lim = 4500
            gov_ratio = 0.65
            if value == 1: #EPS
                if "Sloped roof" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 3 
                    df["roofs_u"] = 0.365
                elif "Horizontal roo" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 0
                    df["roofs_u"] = 0.365
                if roof_eps_cost*gov_ratio <= lim:
                    cost = np.append(cost, roof_eps_cost*(1-gov_ratio))
                elif roof_eps_cost*gov_ratio > lim:
                    cost = np.append(cost, roof_eps_cost-lim)

            elif value == 2: #XPS
                if "Sloped roof" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 4 
                    df["roofs_u"] = 0.326
                elif "Horizontal roof" in str(full_user_data["roof_type"].iloc[0]):
                    df["roofs_type"] = 1
                    df["roofs_u"] = 0.326
                if roof_xps_cost*gov_ratio <= lim:
                    cost = np.append(cost, roof_xps_cost*(1-gov_ratio))
                elif roof_xps_cost*gov_ratio > lim:
                    cost = np.append(roof_xps_cost-lim)
                
        #Floors retrofits
        elif label == "Floor":
            if value == 1:
                floors_eps_cost = 13.5*df["floors_area"].iloc[0]
                lim = 4500
                gov_ratio = 0.65
                if floors_eps_cost*gov_ratio <= lim:
                    cost = np.append(cost, floors_eps_cost*(1-gov_ratio))
                elif floors_eps_cost*gov_ratio > lim:
                    cost = np.append(floors_eps_cost-lim)
                if "Ground slab" in str(full_user_data["floor_type"].iloc[0]):
                    df["floors_type"] = 2
                    df["floors_u"] = 0.32
                elif "Interior floor" in str(full_user_data["floor_type"].iloc[0]):
                    df["floors_type"] = 4
                    df["floors_u"] = 0.32
                else:
                    df["floors_type"] = 0
                    df["floors_u"] = 0.32
                    
        #windows retrofits
        elif label == "Glazing":
            window_alu_cost = 380*df["window_area"].iloc[0]
            window_pvc_cost = 260*df["window_area"].iloc[0]
            gov_ratio = 0.85
            lim = 1500
            if value == 1: #Corte térmico alumínio, duplo
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
        
        elif label == "DHW":

            if value == 1: #Esquentador
                if df["dhw_equipment"].iloc[0] != 1:
                    esq_cost = 450 #per unit
                    df["dhw_equipment"] = 1
                    df["nr_dhw_units"] = 1
                    cost = np.append(cost, esq_cost)
            elif value == 2: #Termoacumulador
                if df["dhw_equipment"].iloc[0] != 8:
                    ter_cost = 175
                    df["dhw_equipment"] = 8
                    df["nr_dhw_units"] = 1
                    cost = np.append(cost, ter_cost)
            elif value == 3: #Caldeira
                if df["dhw_equipment"].iloc[0] != 0:
                    cal_cost = 1750
                    df["dhw_equipment"] = 0
                    df["nr_dhw_units"] = 1
                    cost = np.append(cost, cal_cost)
            elif value == 4: #Bomba de calor - multi split-
                if df["dhw_equipment"].iloc[0] != 2:
                    bom_cost = 3750
                    gov_ratio = 0.85
                    lim=2500
                    df["dhw_equipment"] = 2
                    df["nr_dhw_units"] = 1
                cost = np.append(cost, bom_cost-lim)
            elif value == 5:#Painel Solar
                if df["dhw_equipment"].iloc[0] != 4:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["dhw_equipment"] = 4
                    df["dhw_source"] = 10
                    if df["Tipologia"].iloc[0] <= 3:
                        df["nr_dhw_units"] = 3
                        cost = np.append(cost, sol_cost_3-lim)
                    elif df["Tipologia"].iloc[0] > 3:
                        df["nr_dhw_units"] = 6
                        cost = np.append(cost, sol_cost_6-lim)

        #AC types
        
                    
        elif label == "HVAC":
            if df["dhw_equipment"].iloc[0] == 2:
                df["ac_equipment"] = 4
                df["nr_ac_units"] = df["Tipologia"]
                cost = np.append(cost, 300*df["Tipologia"])    
                
            elif value == 1: #Esquentador
                cost = np.append(cost, 300*df["Tipologia"])
                if df["ac_equipment"].iloc[0] != 3:
                    esq_cost = 450 #per unit
                    df["ac_equipment"] = 3
                    df["nr_ac_units"] = 1
                    cost = np.append(cost, esq_cost)
                
            elif value == 2: #Termoacumulador
                cost = np.append(cost, 300*df["Tipologia"])
                if df["ac_equipment"].iloc[0] != 11 and df["dhw_equipment"].iloc[0] != 8:
                    ter_cost = 175
                    df["ac_equipment"] = 14
                    df["dhw_equipment"] = 8
                    df["nr_ac_units"] = 1
                    cost = np.append(cost, ter_cost)
    
            elif value == 3:#Caldeira Mural + Radiadores fixos
                if df["ac_equipment"].iloc[0] != 7:
                    cal_cost = 2250
                    rad_cost = 15
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_equipment"] = 7
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
                if df["ac_equipment"].iloc[0] != 4:
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_equipment"] = 4
                    df["nr_ac_units"] = df["Tipologia"].iloc[0]
                    if 300*df["Tipologia"].iloc[0]*0.85 <= lim:
                        cost = np.append(cost, 300*df["Tipologia"].iloc[0]*gov_ratio)
                    elif 300*df["Tipologia"]*0.85 > lim:
                        cost = np.append(cost, 300*df["Tipologia"].iloc[0]-300*df["Tipologia"].iloc[0]*gov_ratio)
                        
            elif value == 5:#Painel Solar
                if df["ac_equipment"].iloc[0] != 6 and df["dhw_equipment"].iloc[0] != 4:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_equipment"] = 7
                    df["ac_source"] = 10
                    if df["Tipologia"].iloc[0] <= 3:
                        df["nr_ac_units"] = 3
                        if df["dhw_equipment"].iloc[0] != 5:
                            cost = np.append(cost, sol_cost_3-lim)
                    elif df["Tipologia"].iloc[0] > 3:
                        df["nr_ac_units"] = 6
                        if df["dhw_equipment"].iloc[0] != 5:
                            cost = np.append(cost, sol_cost_6-lim)

                
        #AQS source
        
        elif label == "DHW energy source":
            if value == 1:
                if df["dhw_source"].iloc[0] != 10:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["dhw_source"] = 10
                    if df["dhw_equipment"].iloc[0] != 4 and df["ac_equipment"].iloc[0] != 6:
                        
                        if df["Tipologia"].iloc[0] <= 3:
                            cost = np.append(cost, sol_cost_3-lim)
                        elif df["Tipologia"].iloc[0] > 3:
                            cost = np.append(cost, sol_cost_6-lim)
                
        #AC source
        
        elif label == "HVAC energy source":
            if value == 1:
                if df["ac_source"].iloc[0] != 10:
                    sol_cost_3 = 6100
                    sol_cost_6 = 9400
                    gov_ratio = 0.85
                    lim = 2500
                    df["ac_source"] = 10
                    if df["dhw_equipment"].iloc[0] != 4 and df["ac_equipment"].iloc[0] != 6 and df["dhw_source"].iloc[0] != 10:
                        if df["Tipologia"].iloc[0] <= 3:
                            cost = np.append(cost, sol_cost_3-lim)
                        elif df["Tipologia"].iloc[0] > 3:
                            cost = np.append(cost, sol_cost_6-lim)
    df["cost"] = np.sum(cost).ravel()
    return df


# In[ ]:





# In[176]:


def epc_opt(x):
    epc = model_inputs.copy()
    new_epc = epc.copy()
    final_epc = retrofits(new_epc, x, problem_types_label)
    area_calc = final_epc["Área útil de Pavimento"].iloc[0]
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


# In[177]:


def epc_r(x):
    epc = model_inputs
    new_epc = epc.copy()
    final_epc = retrofits(new_epc, x, problem_types_label)
    area_calc = final_epc["Área útil de Pavimento"].iloc[0]
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


# In[ ]:


if start_opt:
    with st.spinner("""Performing building/house rehabilitation optimization..."""):
        problem = Problem(len(problem_types_label), 3)
        problem.types[:] = problem_types
        problem.function = epc_opt
        # fname =  "NSGAII" + time_str() + ".csv"
        # with open(fname, 'a'):
        algorithm = NSGAII(problem, population_size=25)
        algorithm.run(250)


# In[ ]:


if start_opt:
    x = [s.objectives[0] for s in algorithm.result]
    y = [s.objectives[1] for s in algorithm.result]
    z = [s.objectives[2] for s in algorithm.result]
    results_df = pd.DataFrame([x, y, z]).transpose()
    results_df.columns = ["Ntc [kWh]", "ROI in 3 years [ratio]", "Retrofit cost [€]"]
    results_df["ROI in 3 years [ratio]"] = results_df["ROI in 3 years [ratio]"].apply(lambda x: -x)
    


# In[ ]:





# In[ ]:


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


# In[ ]:


if start_opt:
    sol = []
    for s in algorithm.result:
        for v in enumerate(problem_types):
            int_i = s.variables[v[0]]
            sol = np.hstack((sol, v[1].decode(int_i)))
        vars_opt = pd.DataFrame(sol.reshape(-1, len(problem_types)))
        vars_opt.columns = problem_types_label
        
        full_opt_df = pd.concat([vars_opt, results_df], axis=1).drop_duplicates()
        budget_max_df = full_opt_df[(full_opt_df["Retrofit cost [€]"] <= budget) & 
                            (full_opt_df["ROI in 3 years [ratio]"] > 0)].drop_duplicates(full_opt_df.columns[: len(full_opt_df.columns)-3])
        new_rs = []
        for i, j in enumerate(budget_max_df.index):
            new_rs = np.append(new_rs, epc_r(budget_max_df.drop(["Ntc [kWh]", "ROI in 3 years [ratio]", "Retrofit cost [€]"], axis=1).iloc[i]))
        budget_max_df["New R ratios"] = new_rs
        budget_max_df["New R ratios"] = budget_max_df["New R ratios"].apply(lambda x: r_to_epc(x))
        


# In[ ]:


def retrofit_translate(df1):
    #walls
    df = df1.copy()
    for i in df.reset_index().index:
        for label in problem_types_label:
            if label == "Walls":
                if df["Walls"].iloc[i] == 1:
                    df["Walls"].iloc[i] = "ETICS"
                else:
                    df["Walls"].iloc[i]  = "-"
                    
            #roofs    
            elif label == "Roof":
                if df["Roof"].iloc[i]  == 1:
                    df["Roof"].iloc[i]  = "EPS"
                elif df["Roof"].iloc[i]  == 2:
                    df["Roof"].iloc[i]  = "XPS"
                else:
                    df["Roof"].iloc[i]  = "-"
    
            #floors
            elif label == "Floor":
                if df["Floor"].iloc[i]  == 1:
                    df["Floor"].iloc[i]  = "EPS"
                else:
                    df["Floor"].iloc[i]  = "-"
                    
            #windows    
            elif label == "Glazing":
                if df["Glazing"].iloc[i]  == 1:
                    df["Glazing"].iloc[i]  = "PVC/double glazing"
                elif df["Glazing"].iloc[i]  == 2:
                    df["Glazing"].iloc[i]  = "Metal frame w/ thermal cut/double glazing"
                else:
                    df["Glazing"].iloc[i]  = "-"
            
            #DHW_t
            elif label == "DHW":
                if df["DHW"].iloc[i] == 1 and model_inputs["dhw_equipment"].iloc[0] != 1:
                    df["DHW"].iloc[i] = "Heater"
                elif df["DHW"].iloc[i] == 2 and model_inputs["dhw_equipment"].iloc[0] != 8:
                    df["DHW"].iloc[i] = "Water heater"
                elif df["DHW"].iloc[i] == 3 and model_inputs["dhw_equipment"].iloc[0] != 0:
                    df["DHW"].iloc[i] = "Boiler"
                elif df["DHW"].iloc[i] == 4 and model_inputs["dhw_equipment"].iloc[0] != 2:
                    df["DHW"].iloc[i] = "Heat recovery"
                elif df["DHW"].iloc[i] == 5 and model_inputs["dhw_equipment"].iloc[0] != 4:
                    df["DHW"].iloc[i] = "Solar Panels"
                else:
                    df["DHW"].iloc[i] = "-"
            
            #HVAC_t
            elif label == "HVAC":
                if df["DHW"].iloc[i] == 2  and model_inputs["dhw_equipment"].iloc[0] != 2:
                    df["HVAC"].iloc[i] = "Air-Water heat pump"
                    df["DHW"].iloc[i] = "Air-Water heat pump"
                else:
                    if df["HVAC"].iloc[i] == 1 and model_inputs["ac_equipment"].iloc[0] != 3:
                        df["HVAC"].iloc[i] = "Heater with radiators"
                    elif df["HVAC"].iloc[i] == 2 and model_inputs["ac_equipment"].iloc[0] != 11:
                        df["HVAC"].iloc[i] = "Water heater with radiators"
                    elif df["HVAC"].iloc[i] == 3 and model_inputs["ac_equipment"].iloc[0] != 1:
                        df["HVAC"].iloc[i] = "Boiler with radiators"
                    elif df["HVAC"].iloc[i] == 4 and model_inputs["ac_equipment"].iloc[0] != 4:
                        df["HVAC"].iloc[i] = "Multi-split"
                    elif df["HVAC"].iloc[i] == 5 and model_inputs["ac_equipment"].iloc[0] != 6:
                        df["HVAC"].iloc[i] = "Solar panels"
                    else:
                        df["HVAC"].iloc[i] = "-"
            
            #DHW_s
            elif label == "DHW energy source":
                if df["DHW energy source"].iloc[i] == 1 and model_inputs["dhw_source"].iloc[0] != 10:
                    df["DHW energy source"].iloc[i] = "Solar"
                else:
                    df["DHW energy source"].iloc[i] = "-"
                    
            
            #HVAC_s
            elif label == "HVAC energy source":
                if df["HVAC energy source"].iloc[i] == 1 and model_inputs["HVAC_source"].iloc[0] != 10:
                    df["HVAC energy source"].iloc[i] = "Solar"
                else:
                    df["HVAC energy source"].iloc[i] = "-"

    return df


# In[ ]:


if start_opt:
    new_r = []
    for i  in vars_opt.index:
        new_r = np.append(new_r, epc_r(vars_opt.iloc[i]))
    results_df["New R ratios"] = new_r
    results_df["New R ratios"] = results_df["New R ratios"]


# In[ ]:


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# In[ ]:


# import plotly.express as px
# if start_opt:
#     fig = px.scatter_3d(results_df, 
#                         x="Ntc [kWh]", 
#                         y="ROI in 3 years [ratio]", 
#                         z="Retrofit cost [€]",
#                         color="New R ratios",
#                         color_continuous_scale=px.colors.diverging.Tealrose)

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


if start_opt:
    csv_df = budget_max_df.copy()
#     col51, col52, col53 = st.columns(3)
    csv_df.index = ["optimum solution " + str(i+1) for i, j in enumerate(csv_df.index)]
    string_table = retrofit_translate(csv_df)
    categories_pl = retrofit_translate(csv_df)
    csv = convert_df(string_table)
    categories_pl
    


# In[ ]:


if start_opt:
    chart = categories_pl[["Ntc [kWh]", "ROI in 3 years [ratio]", "Retrofit cost [€]"]].reset_index(drop=True)
    bar_idx = []
    for i, j in enumerate(chart.index):

        bar_idx = np.append(bar_idx, "optimum solution " + str(i + 1))
    chart["solution"] = bar_idx
    bar_chart = chart.melt(id_vars="solution")


# In[ ]:


if start_opt:
    import plotly.express as px
    fig = px.bar(bar_chart[bar_chart["variable"] == "Ntc [kWh]"], x="variable", y="value", barmode="group", color="solution", color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    #fig.show()


# In[ ]:


if start_opt:
    import plotly.express as px
    fig = px.bar(bar_chart[bar_chart["variable"] == "ROI in 3 years [ratio]"], x="variable", y="value", barmode="group", color="solution", color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    #fig.show()


# In[ ]:


if start_opt:
    import plotly.express as px
    fig = px.bar(bar_chart[bar_chart["variable"] == "Retrofit cost [€]"], x="variable", y="value", barmode="group", color="solution", color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig)
    #fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




