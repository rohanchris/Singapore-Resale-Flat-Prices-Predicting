# Singapore-Resale-Flat-Prices-Predicting
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import sklearn
from PIL import Image

pandas (pd): A library used for data manipulation and analysis, especially for working with data in DataFrame format.
numpy (np): A library used for numerical computations in Python, especially for working with arrays.
warnings: The filterwarnings function is used to ignore warning messages that can clutter the output.
datetime: A module to handle dates and times in Python.
streamlit (st): A framework to build interactive web applications, particularly for data science and machine learning models.
streamlit_option_menu: A module used to create an option menu in a Streamlit app. It helps in creating navigable options in the sidebar.
pickle: A module used to serialize and deserialize Python objects, often used to save and load machine learning models.
sklearn: A library for machine learning in Python, which includes various tools for model building, data preprocessing, and evaluation.
PIL (Image): A module for working with images in Python, allowing you to open, manipulate, and save different image file formats.
2. Purpose of the Code
This code appears to be setting up an environment for a machine learning or data analysis web application using Streamlit.
The libraries imported are commonly used for data manipulation (pandas), numerical computations (numpy), working with dates (datetime), building interactive web apps (streamlit), and handling machine learning models (pickle and sklearn).
The PIL library suggests that the application may involve image processing or displaying images.
The streamlit_option_menu is likely used to create a user-friendly interface for navigating different sections of the app.
3. What to Expect Next
After setting up these imports, the code might include data loading, model loading (via pickle), building an interactive user interface with Streamlit, and possibly some image processing or visualization tasks.
The main body of the Streamlit app would likely include widgets like buttons, sliders, and menus to interact with the data or machine learning models.

Code Explanation
*Function Definition:
def town_mapping(town_map):
The function town_mapping is defined with a single parameter, town_map, which is expected to be a string representing a town name.

*Mapping Town Names to Integers:
if town_map == 'ANG MO KIO':
    town_1 = int(0)
elif town_map == 'BEDOK':
    town_1 = int(1)
# ... other elif statements ...
return town_1
The function uses a series of if-elif statements to compare the input town_map against known town names.
For each town name, a specific integer is assigned to town_1. For example:
'ANG MO KIO' is mapped to 0.
'BEDOK' is mapped to 1.
This pattern continues for all the towns listed in the function.

*Returning the Mapped Integer:
return town_1
After finding the matching town name, the function returns the corresponding integer value (town_1).

*Potential Issue:
There are a couple of typos in the elif conditions:
In some cases, the variable town is used instead of town_map. Specifically:
elif town == 'PASIR RIS':
elif town == 'PUNGGOL':
# ... similar mistakes ...
These lines should use town_map instead of town to maintain consistency.

If you call the function like this:
town_mapping('BEDOK')
The function would return 1 because 'BEDOK' is mapped to 1 in the series of conditions.

1. flat_type_mapping(flt_type)
Purpose: Maps different flat types to specific integer values.
Logic:
if flt_type == '3 ROOM':
    flat_type_1= int(2)
elif flt_type == '4 ROOM':
    flat_type_1= int(3)
# ... other elif statements ...
return flat_type_1
The function compares the input flt_type (a string representing a flat type) to a set of predefined flat types.
If a match is found, it assigns a corresponding integer value to the variable flat_type_1.
For example:
'3 ROOM' is mapped to 2.
'4 ROOM' is mapped to 3.
'1 ROOM' is mapped to 0.
The function then returns the mapped integer value.

flat_type_mapping('4 ROOM')
This would return 3 because '4 ROOM' is mapped to 3 in the function.

2. flat_model_mapping(fl_m)
Purpose: Maps different flat models to specific integer values.
Logic:
2. flat_model_mapping(fl_m)
Purpose: Maps different flat models to specific integer values.
Logic:
if fl_m == 'Improved':
    flat_model_1= int(5)
elif fl_m == 'New Generation':
    flat_model_1= int(12)
# ... other elif statements ...
return flat_model_1
Similar to the flat_type_mapping function, this function compares the input fl_m (a string representing a flat model) to a set of predefined flat models.
If a match is found, it assigns a corresponding integer value to the variable flat_model_1.
For example:
'Improved' is mapped to 5.
'New Generation' is mapped to 12.
'2-room' is mapped to 0.
The function then returns the mapped integer value.

flat_model_mapping('Standard')
This would return 17 because 'Standard' is mapped to 17 in the function.

1. Predict Price Function
python
Copy code
def predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt):
Inputs:
year: Year of the transaction.
town: Town name where the flat is located.
flat_type: Type of the flat (e.g., 3 ROOM, 4 ROOM).
flr_area_sqm: Floor area of the flat in square meters.
flat_model: The model/type of the flat.
stry_start, stry_end: The range of storeys (floors) where the flat is located.
re_les_year, re_les_month: Remaining lease years and months.
les_coms_dt: The lease commencement date.

Processing:

Converts and maps categorical inputs (like town and flat_type) to numerical values using predefined mapping functions (town_mapping, flat_type_mapping, and flat_model_mapping).
Logarithmically transforms storey start and end values to normalize them.
Combines all the input features into a NumPy array called user_data.
Loads a pre-trained model from a pickle file (Resale_Flat_Prices_Model_1.pkl).
Uses the model to predict the log-transformed price, then exponentiates it to obtain the actual predicted price.
Output:

The predicted price, rounded to the nearest integer.

2. Streamlit App Layout
python
Copy code
st.set_page_config(layout="wide")
st.title("SINGAPORE RESALE FLAT PRICES PREDICTING")
Sidebar Menu:

Uses the option_menu to create a navigable sidebar with options: Home, Price Prediction, and About.
Home Section:

Displays an image and information about HDB flats, the resale process, valuation, eligibility criteria, and market trends.
Price Prediction Section:

Allows users to input various flat details (e.g., year, town, flat type, floor area, storey range, etc.).
A button triggers the predict_price function with the inputs and displays the predicted price.
About Section:

Provides detailed information about the data collection, feature engineering, model selection, evaluation, and deployment process of the application.

3. Interactive Elements in Streamlit
st.selectbox: Dropdown menus for users to select options (e.g., year, town, flat type).
st.number_input: Numeric input fields for entering values like floor area, storey range, and remaining lease.
st.button: A button that, when clicked, triggers the price prediction.
st.write: Displays the predicted price and additional information.

4. Model Usage
The application loads a machine learning model from a pickle file to make predictions based on user input. The model is assumed to be trained on historical resale flat transaction data.

Summarry
1. Function Definitions for Mapping:
town_mapping(town_map): Converts town names into corresponding integer values.
flat_type_mapping(flt_type): Maps different flat types (e.g., '3 ROOM', '4 ROOM') to specific integers.
flat_model_mapping(fl_m): Maps various flat models (e.g., 'Improved', 'New Generation') to corresponding integers.
These mappings are used to convert categorical data into numerical format, which is essential for machine learning models.

2. Price Prediction Function:
predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt):
Takes various details about an HDB flat (e.g., year of transaction, town, flat type, floor area, flat model, storey range, remaining lease) as inputs.
Converts and processes these inputs using the previously defined mapping functions and logarithmic transformations.
Loads a pre-trained machine learning model (Resale_Flat_Prices_Model_1.pkl) to predict the resale price.
Returns the predicted price after reversing any log transformations.

3. Streamlit Web Application:
App Layout:

The application is set up using Streamlit with a wide layout and a title.
A sidebar menu allows navigation between "Home," "Price Prediction," and "About" sections.
Home Section:

Displays an image and provides detailed information about HDB flats, the resale process, valuation, eligibility criteria, and market trends.
Price Prediction Section:

Interactive elements (dropdowns, numeric inputs) allow users to enter details about a flat.
A button triggers the price prediction function, which displays the predicted price to the user.
About Section:

Describes the process of data collection, feature engineering, model training, evaluation, and the development of the web application.

4. Model Usage:
The app uses a pre-trained machine learning model to predict resale prices based on user input. This model is assumed to have been trained on historical data of resale flat transactions.
This code builds a comprehensive web application that allows users to predict the resale price of HDB flats in Singapore. The app takes user inputs, processes them through mapping functions, and uses a trained machine learning model to provide predictions. It also educates users about the resale process and the factors affecting flat prices. The application is designed for deployment, making it accessible to users online.
