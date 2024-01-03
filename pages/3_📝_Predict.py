import streamlit as st 
import pandas as pd 
import joblib



with st.sidebar:
    st.error(" ❌The model is predicting very extreme values after getting input features from user.")
    st.warning("⛏️We are working on this.")
#Loading the mode 
loaded_model = joblib.load("Model_Poly.pkl")
if loaded_model: 
    st.success("The model is loaded successfully")

# Creating a Dataframe to feed the saved machine learning model 

features = ['Item_MRP','Item_Visibility','Item_Weight','Item_Fat_Content','Outlet_Age','Outlet_Establishment_Year','Outlet_Size','Outlet_Type']
user_data = pd.DataFrame(columns=features)




st.header("Please enter your features values")

# ... (previous code)

# Feature 1 
# Item MRP 
st.subheader("Item MRP")
MRP = st.number_input(label="Enter MRP...", placeholder='Type a number')
user_data.at[0, 'Item_MRP'] = MRP

# Feature 2 
# Item Visibility 
st.subheader("Item Visibility")
item_visibility = st.slider(
    "Item Visibility",
    min_value=0.0,
    max_value=0.45,
    value=0.2,  # Optional default value
    step=0.01,
    format="%.2f"
)
user_data.at[0, 'Item_Visibility'] = item_visibility

# Feature 3 
# Weight 
st.subheader("Weight")
weight = st.number_input(label="Enter weight...", placeholder='Type a number')
user_data.at[0, 'Item_Weight'] = weight

# Feature 4 
# Item Fat Content 
st.subheader("Fat Content")
user_fat_content = st.selectbox("Fat Content", ["Low", "Regular", "Non-Edible"])
if user_fat_content == "Regular":
    user_data.at[0, 'Item_Fat_Content'] = 0
elif user_fat_content == "Low":
    user_data.at[0, 'Item_Fat_Content'] = 1
else:  # if Non - Edible
    user_data.at[0, 'Item_Fat_Content'] = 2

# Feature 5 
# Outlet Age 
age = st.number_input(label="Enter Age...", placeholder='Type a number')
user_data.at[0, 'Outlet_Age'] = age

# Feature 6 
# Outlet Established Year 
outlet_establishment_year = st.number_input(
    label="Enter the establishment year (1985-2009):",
    min_value=1985,
    max_value=2009,
    value=2000,  # Optional default value
    step=1
)
if outlet_establishment_year:
    user_data.at[0, 'Outlet_Establishment_Year'] = outlet_establishment_year

# Feature 7 
# Outlet Size 
st.subheader("Outlet Size")
user_outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "Large"])
if user_outlet_size == "Small":
    user_data.at[0, 'Outlet_Size'] = 0
elif user_outlet_size == "Medium":
    user_data.at[0, 'Outlet_Size'] = 1
else:  # if Large
    user_data.at[0, 'Outlet_Size'] = 2

# Feature 8 
# Outlet Type
st.subheader("Outlet Type")
user_outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "SuperMarket Type1", "SuperMarket Type2", "SuperMarket Type3"])
if user_outlet_type == "Grocery Store":
    user_data.at[0, 'Outlet_Type'] = 0
elif user_outlet_type == 'SuperMarket Type1':
    user_data.at[0, 'Outlet_Type'] = 1
elif user_outlet_type == 'SuperMarket Type2':
    user_data.at[0, 'Outlet_Type'] = 2
elif user_outlet_type == 'SuperMarket Type3':
    user_data.at[0, 'Outlet_Type'] = 3

# ... (rest of your code)


st.write(user_data)

#Loading the MinMax Scalar
mm = joblib.load('Scalar_Poly.pkl')
user_data_scaled = mm.transform(user_data)

st.header("Prediction")
if st.button("Predict"):
    try:
        if not user_data.empty:  # Check if the DataFrame is not empty
            # Make prediction
            prediction = loaded_model.predict(user_data_scaled)

            # Display prediction
            st.success(prediction)
        else:
            st.warning("Please fill in the input features before predicting.")
    except Exception as e:
        st.error("An error occurred: {}".format(e))



