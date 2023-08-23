import numpy as np
import pickle
import streamlit as st
import math
from sklearn.preprocessing import StandardScaler


loaded_model = pickle.load(open(r'C:\Users\lenovo\Desktop\Data Science Final\projects\Life_expectancy_prediction\Life_expectancy_model.pkl', 'rb'))

scaler = StandardScaler()
with open(r'C:\Users\lenovo\Desktop\Data Science Final\projects\Life_expectancy_prediction\fitted_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



def life_expectancy_prediction(input):
    input_array = np.asarray(input)
    input_reshaped = input_array.reshape(1, -1)
    # input_reshaped[0,3] = math.log(input_reshaped[0,3])
    input_standardized = scaler.transform(input_reshaped)
    prediction = loaded_model.predict(input_standardized)
    return prediction



def main():
    # giving a title
    st.title('Life Expectancy Prediction App')

    # getting the input data from the user

    Schooling = st.text_input('Schooling (Number of years of Schooling):')
    Adult_Mortality = st.text_input('Adult Mortality (per 1000 population):')
    Under_five_deaths= st.text_input('Under Five Deaths (per 1000 population):')

    schooling_range = (0,20)
    adult_mortality_range = (0, 1000)
    under_five_deaths_range = (0, 1000)



    diagnosis = ""

    # creating a button for Prediction

    if st.button('Life Expectancy'):
        if (float(Schooling) < schooling_range[0] or float(Schooling) > schooling_range[1] or
                float(Adult_Mortality) < adult_mortality_range[0] or float(Adult_Mortality) > adult_mortality_range[
                    1] or
                float(Under_five_deaths) < under_five_deaths_range[0] or float(Under_five_deaths) >
                under_five_deaths_range[1]):
            st.warning('Please enter valid inputs within the specified ranges.')
        else:
            diagnosis = life_expectancy_prediction([float(Schooling),float(Adult_Mortality),float(Under_five_deaths)])

            st.success(diagnosis)


if __name__ == '__main__':
    main()

