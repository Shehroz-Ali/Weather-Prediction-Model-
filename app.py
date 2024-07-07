import streamlit as st
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
import pickle
# Load the trained model
# Set the layout to wide

st.set_page_config(layout="wide")

def func():

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return scaler ,model

# Weather class labels and corresponding video URLs
class_labels = {
    2: {
        'label': 'Partly Cloudy',
        'video_url': 'pc.mp4'
    },
    0: {
        'label': 'Clear',
        'video_url': 'clear.mp4'
    },
    1: {
        'label': 'Foggy',
        'video_url': 'foggy.mp4'
    }
}



# Create the Streamlit app

st.title('Weather Prediction App')
st.markdown('This app predicts the weather class and displays a video based on the prediction.')
st.markdown('---')

# Add a description of your project
st.header('Project Description')
st.markdown('''The Weather Prediction App is a web application built using Streamlit. Its main purpose is to predict the weather class based on user-provided input and display a corresponding video representing the predicted weather condition.

The app utilizes a trained machine learning model that has been pre-loaded using Pickle .And has an accuracy of 95%. The model takes into account various weather features such as temperature, humidity, wind speed, wind bearing, visibility, pressure, day, month, hour, and precipitation. These features are used to make predictions on the weather class.

Upon launching the app, users are presented with a user-friendly interface. The main section of the app displays a project description, providing an overview of the app's functionality. In the sidebar, users can find a set of input fields corresponding to the weather features. They can input values for each feature by either typing in numerical values or using a slider, depending on the specific input field.

Once the user has provided values for all the weather features, they can click the "Predict" button to trigger the prediction process. The input data is then preprocessed using a scaler, and the preprocessed data is fed into the trained model. The model predicts the weather class based on the input, and the predicted class label is displayed on the app's interface.

Furthermore, the app also dynamically displays a video related to the predicted weather class. Each weather class is associated with a specific video, and the corresponding video is shown to the user. This visual representation adds an interactive and engaging element to the app, providing users with a more immersive experience.

Overall, the Weather Prediction App offers a simple and intuitive way for users to input weather features and receive real-time predictions on the weather class. It enhances user understanding and engagement by visually representing the predicted weather condition through videos, making it an informative and enjoyable tool for weather enthusiasts and individuals interested in exploring weather patterns.''')

st.markdown('---')
            

# Create input form for user input
logo_image = 'logo.jpg'
st.sidebar.image(logo_image, width=100, caption='Weatherly', use_column_width=False, output_format='JPG')

st.sidebar.title('\u2601 \u26A1 \u26C8 \u2600 \u2744 \u2614')
st.sidebar.subheader('Weather Features')
temperature = st.sidebar.number_input('Temperature (C)', step=0.1)
humidity = st.sidebar.number_input('Humidity', step=0.1)
wind_speed = st.sidebar.number_input('Wind Speed (km/h)', step=0.1)
wind_bearing = st.sidebar.number_input('Wind Bearing (degrees)', step=0.1)
visibility = st.sidebar.number_input('Visibility (km)', step=0.1)
pressure = st.sidebar.number_input('Pressure (millibars)', step=0.1)
day = st.sidebar.number_input('Day', step=0.1)
month = st.sidebar.number_input('Month', step=0.1)
hour = st.sidebar.number_input('Hour', step=0.1)
precipitation = (temperature * humidity * pressure) / 10000

feature_names = [ 'Temperature (C)', 'Humidity',
                 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                 'Pressure (millibars)', 'day', 'month', 'hour', 'Precipitation']


if st.sidebar.button('Predict'):
    if any([not temperature, not humidity, not wind_speed, not wind_bearing, not visibility, not pressure, not day, not month, not hour]):
        st.warning('Please fill in all input fields.')
    else:
        scaler,model=func()
        input_data = pd.DataFrame([[temperature, humidity, wind_speed, wind_bearing, visibility, pressure, day, month, hour, precipitation]], columns=feature_names)
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_data_scaled)[0]

        # Display the prediction result
        
        lab = class_labels[prediction]['label']
        
        st.header(f'Prediction: {lab}')

        # Display the video based on the predicted weather class
        video_url = class_labels[prediction]['video_url']
        st.video(video_url)
##        video_html = f'''
##        <video width="100%" height="auto" loop autoplay controls>
##            <source src="{video_url}" type="video/mp4">
##        </video>
##        '''
##        st.markdown(video_html, unsafe_allow_html=True)




