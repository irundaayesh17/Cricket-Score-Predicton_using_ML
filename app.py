import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained pipeline model
TrainedModel_GBoosterRegression = pickle.load(open('t20_score_predictor_GBoostRegression.pkl', 'rb'))
TrainedModel_XGBooster = pickle.load(open('t20_score_predictor_xgbooster.pkl', 'rb'))
TrainedModel_ANNModel = pickle.load(open('t20_score_predictor_ann_model.pkl', 'rb'))


# List of venues from your provided list
venues = ['Melbourne Cricket Ground', 'Simonds Stadium, South Geelong', 'Adelaide Oval', 'McLean Park', 'Bay Oval',
          'Eden Park', 'The Rose Bowl', 'County Ground', 'Sophia Gardens', 'Riverside Ground', 'Green Park',
          'Vidarbha Cricket Association Stadium, Jamtha', 'M Chinnaswamy Stadium', 'Central Broward Regional Park Stadium Turf Ground',
          'Dubai International Cricket Stadium', 'Sheikh Zayed Stadium', 'Sydney Cricket Ground', 'Bellerive Oval',
          'Westpac Stadium', 'Seddon Park', 'Mangaung Oval', 'Senwes Park', 'Kensington Oval, Bridgetown',
          "Queen's Park Oval, Port of Spain", 'R Premadasa Stadium', 'Warner Park, Basseterre', 'Sabina Park, Kingston',
          'R.Premadasa Stadium, Khettarama', 'Saxton Oval', 'JSCA International Stadium Complex', 'Edgbaston', 'Old Trafford',
          'Arun Jaitley Stadium', 'Saurashtra Cricket Association Stadium', 'Greenfield International Stadium', 'Gaddafi Stadium',
          'The Wanderers Stadium', 'SuperSport Park', 'Newlands', 'Barabati Stadium', 'Holkar Cricket Stadium', 'Wankhede Stadium',
          'Shere Bangla National Stadium, Mirpur', 'Sylhet International Cricket Stadium', 'National Stadium', 'Harare Sports Club',
          'Carrara Oval', 'Brisbane Cricket Ground, Woolloongabba', 'Rajiv Gandhi International Cricket Stadium', 'Eden Gardens',
          'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium', 'MA Chidambaram Stadium, Chepauk', 'Darren Sammy National Cricket Stadium, St Lucia',
          'Warner Park, St Kitts', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'M.Chinnaswamy Stadium', 'Manuka Oval', 
          'Perth Stadium', 'Buffalo Park', 'Kingsmead', 'St George\'s Park', 'Punjab Cricket Association IS Bindra Stadium, Mohali',
          'Rajiv Gandhi International Stadium, Uppal', 'Hagley Oval', 'Providence Stadium, Guyana', 'Pallekele International Cricket Stadium',
          'Zahur Ahmed Chowdhury Stadium', 'Maharashtra Cricket Association Stadium', 'Boland Park', 'New Wanderers Stadium',
          'Kennington Oval', 'Western Australia Cricket Association Ground', 'Brabourne Stadium', 'Jade Stadium', 'Gymkhana Club Ground',
          'Trent Bridge', 'Lord\'s', 'Maple Leaf North-West Ground', 'AMI Stadium', 'Providence Stadium', 'Beausejour Stadium, Gros Islet',
          'Punjab Cricket Association Stadium, Mohali', 'Sir Vivian Richards Stadium, North Sound', 'Moses Mabhida Stadium', 'Stadium Australia',
          'Shere Bangla National Stadium', 'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa', 'Subrata Roy Sahara Stadium',
          'Sardar Patel Stadium, Motera', 'Arnos Vale Ground, Kingstown', 'Sharjah Cricket Stadium', 'Windsor Park, Roseau',
          'Himachal Pradesh Cricket Association Stadium', 'Feroz Shah Kotla']

# Define teams
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan',
         'Pakistan', 'Sri Lanka']

# Streamlit UI components
st.title('Cricket Score Predictor')

# Select teams using columns
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# Select venue
venue = st.selectbox('Select venue', sorted(venues))

# Input fields for cricket match data
col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done (works for overs > 5)')
with col5:
    wickets = st.number_input('Wickets out')

# Runs scored in the last 5 overs
last_five = st.number_input('Runs scored in last 5 overs')

model_choice = st.selectbox(
    'Choose the model for prediction',
    ['Gradient Boost Regression', 'XGBoost Regression', 'ANN Model']
)

# Prediction logic
if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'venue': [venue], 'current_score': [current_score],
         'balls_left': [balls_left], 'wickets_left': [wickets_left], 'crr': [crr], 'last_five': [last_five]})

     # Select the model based on user choice
    if model_choice == 'Gradient Boost Regression':
        result = TrainedModel_GBoosterRegression.predict(input_df)
    elif model_choice == 'XGBoost Regression':
        result = TrainedModel_XGBooster.predict(input_df)
    else:
        result = TrainedModel_ANNModel.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))
