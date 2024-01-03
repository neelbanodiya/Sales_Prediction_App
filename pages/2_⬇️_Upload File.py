# Importing necessaries libraries 
import streamlit as st
import pandas as pd
import joblib  # For loading your model
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder



# Load your trained machine learning model
model = joblib.load('Model_Poly.pkl')

#Loading the MinMax Scalar
mm = joblib.load('Scalar_Poly.pkl')

def predict(test_data):
    # Perform necessary data preprocessing
    preprocessed_data = preprocess_data(test_data)


    #Perform data tranformation (Min Max Scalar)
    final_data = scaled(preprocessed_data)
  
    # Make predictions using the model
    predictions = model.predict(final_data)


    return predictions

# Define a function for preprocessing the data 
def preprocess_data(input_data):
    # Make a copy to avoid modifying the original dataset
    df = input_data.copy()

    ## CATEGORICAL DATA PREPROCESSING

    # Creating a broad category for food, drinks & non-comsumables
    df['Item_Categories'] = df['Item_Identifier'].str[0:2]

    # Removing inconsistency from the data
    df['Item_Fat_Content'] = df['Item_Fat_Content'].str.lower().replace({'lf': 'low fat', 'reg': 'regular'})

    # Changing fat content for household & health and hygiene to non-edible
    df.loc[df['Item_Type']=='Household', 'Item_Fat_Content'] = 'Non-Edible'
    df.loc[df['Item_Type']=='Health and Hygiene', 'Item_Fat_Content'] = 'Non-Edible'

    # Grouping 'Starchy Foods' and 'Breakfast' to 'Others'
    df['Item_Type'] = df['Item_Type'].replace(['Starchy Foods', 'Breakfast'], 'Others')

    # Grouping 'Seafood' to 'Meat'
    df['Item_Type'] = df['Item_Type'].replace('Seafood', 'Meat')

    # Grouping 'Breads' to 'Baking goods'
    df['Item_Type'] = df['Item_Type'].replace('Breads', 'Baking Goods')

    # Grouping 'Soft Drinks' and 'Hard Drinks' to 'Drinks'
    df['Item_Type'] = df['Item_Type'].replace(['Soft Drinks', 'Hard Drinks'], 'Drinks')

    # Replacing 'High' to 'Large' in outlet size
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    df['Outlet_Size'] = df['Outlet_Size'].replace('High', 'Large')

    ## NUMERICAL DATA PREPROCESSING

    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

    # Creating a new column "Outlet_Age"
    df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]

    def remove_outliers(col):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        upper_limit = q3 + 1.5*iqr
        count_outliers = df[df[col] > upper_limit].shape[0]

       
        return df[df[col] < upper_limit]

    df = remove_outliers('Item_Visibility')

    df['Item_Visibility'] = np.sqrt(df['Item_Visibility'])

    ## DATA ENCODING

    # Drop 'Item_Identifier' column
    df.drop('Item_Identifier', axis=1, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(data=df, columns=['Item_Type', 'Outlet_Identifier', 'Item_Categories'])

    # Label encoding 'Item_Fat_Content' column
    df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'regular': 0, 'low fat': 1, 'Non-Edible': 2}).astype(int)

    # Label encoding 'Outlet_Size' column
    df['Outlet_Size'] = df['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'Large': 3}).astype(int)

    # Label encoding 'Outlet_Location_Type' and 'Outlet_Type' columns
    le = LabelEncoder()
    df['Outlet_Location_Type'] = le.fit_transform(df['Outlet_Location_Type'])
    df['Outlet_Type'] = le.fit_transform(df['Outlet_Type'])
    
    df = df[['Item_MRP','Item_Visibility','Item_Weight','Item_Fat_Content','Outlet_Age','Outlet_Establishment_Year','Outlet_Size','Outlet_Type']]

   
    return df

# Creating function for applying MinMax Scalar on preprocessed data 

def scaled(data):
   df = mm.transform(data)
   return df
  

# Create the Streamlit app
st.title("Sales Prediction :money_with_wings:")


# Sidbar
with st.sidebar:
  
  st.info("Please upload your CSV file")
  st.warning("⚠️ Please make sure the CSV file don't contain target variable")


# Add a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Call the prediction function
    predictions = predict(df)


    # Display the predictions
    st.write("Predictions:", predictions)
