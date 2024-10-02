import streamlit as st
import joblib
import pandas as pd
import time  # For adding delay to mimic animation

# Load the dataset and model
df_for_training = pd.read_csv(r"C:\Users\Bhuvanesh\Desktop\Yogesh Guvi\Car Dekho\Dataset\Final\Data_after_EDA.csv")
oem_model_mapping = df_for_training.groupby('oem')['model'].unique().to_dict()

km = ['30000-40000', '10000-20000', '60000-70000', '20000-30000',
      '50000-60000', '40000-50000', '100000-110000', '80000-90000',
      '110000-120000', '70000-80000', '0-10000', '90000-100000']
km_sorted = sorted(km)

model_path = r"C:\Users\Bhuvanesh\Desktop\Yogesh Guvi\Car Dekho\Dataset\Final\random_forest_model_reduced.pkl"
model = joblib.load(model_path)

# Define a function to handle user input and convert it to the required format
def user_input_features():
    # Split input fields into two columns
    col1, col2 = st.columns(2)

    with col1:
        oem = st.selectbox('🚗 Select OEM', list(oem_model_mapping.keys()))  # Use keys of the mapping
        model_year = st.slider('📅 Model Year', 2010, 2023)  # Adjusted the range to fit possible years
    
    with col2:
        owner_no = st.selectbox('👤 Owner Number', [1, 2, 3])
        km_bucket = st.selectbox('📏 KiloMeters', km_sorted)

    # Model name will be in a new row (below the first two columns)
    model_name = st.selectbox('🚙 Select Model', oem_model_mapping[oem])  # Filter models based on selected OEM

    # Create a dictionary of user inputs
    data = {
        'oem': oem,
        'modelYear': model_year,
        'ownerNo': owner_no,
        'km_bucket': km_bucket
    }
    features = pd.DataFrame([data])  # Convert to a DataFrame for easy processing
    return features

# Function for encoding and preprocessing
def preprocess_input(df):
    # Apply one-hot encoding using the same encoder that was used for training
    df_encoded = pd.get_dummies(df, drop_first=True)  # Example for one-hot encoding

    # Ensure input has the same columns as training data
    model_columns = model.feature_names_in_  # Get the feature names used during training
    missing_cols = set(model_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0  # Add missing columns with default value

    # Reorder columns to match training set
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    return df_encoded

# Streamlit app main function
def main():
    st.title('🚗 Car Price Prediction')

    # Display the overlay image at the top
    st.image(r"C:\Users\Bhuvanesh\Desktop\Yogesh Guvi\Car Dekho\Dataset\Final\Sample.jpg", use_column_width=True)  # Replace with your image path

    # Get user input
    input_df = user_input_features()

    # Add a submission button for calculating price
    if st.button('💰 Calculate Price'):
        # Add some delay to mimic an animation effect
        with st.spinner('Calculating... 🔄'):
            time.sleep(2)  # Adding a 2-second delay to simulate processing time
        
        # Preprocess the input
        input_processed = preprocess_input(input_df)

        # Make predictions
        prediction = model.predict(input_processed)

        # Convert prediction to lakhs
        prediction_in_lakhs = prediction[0] / 100000

        # Show a congratulatory message with firecracker emoji
        st.subheader('🎉 Your Price Prediction is Ready! 🎉')
        
        # Display the result in lakhs with styled text
        st.markdown(f"<h2 style='color: green;'>💵 ₹ {prediction_in_lakhs:.2f} Lakhs</h2>", unsafe_allow_html=True)

        # Mimic a celebratory pop-up by showing more emojis
        st.balloons()  # This triggers Streamlit's built-in balloon animation

if __name__ == '__main__':
    main()
