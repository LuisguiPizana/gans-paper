import streamlit as st
import pandas as pd
import os
import json

BASE_PATH = "../"

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' - INFO - ')
            #timestamp = parts[0].strip()
            # Replace single quotes with double quotes to make it valid JSON
            log_entry = parts[1].strip().replace("'", '"')
            try:
                # Parse the string into JSON
                log_entry = json.loads(log_entry)
            except json.JSONDecodeError as e:
                continue  # Skip this line if there's an error
            #log_entry['timestamp'] = timestamp
            data.append(log_entry)
    return pd.DataFrame(data)

# Function to load experiments
def load_experiments():
    experiments_path = os.path.join(BASE_PATH, "experiments")
    return os.listdir(experiments_path), experiments_path

# Layout of the dashboard
st.title('Experiment Tracking Dashboard')

# Select an experiment
experiments_list, experiments_path = load_experiments()
selected_experiment = st.selectbox('Select an Experiment', experiments_list)

# Paths for the selected experiment
experiment_path = os.path.join(experiments_path, selected_experiment)
metrics_path = os.path.join(experiment_path, 'metrics_logs.txt')
gradients_path = os.path.join(experiment_path, 'gradients_logs.txt')
is_path = os.path.join(experiment_path, 'inception_score_logs.txt')
samples_path = os.path.join(experiment_path, 'samples')

# Load data
metrics_data = parse_log_file(metrics_path)
gradients_data = parse_log_file(gradients_path)
is_data = parse_log_file(is_path)

# Prepare data
lr_data = metrics_data[['Iteration', 'Discriminator LR', 'Generator LR']]
metrics_data = metrics_data.drop(columns=['Discriminator LR', 'Generator LR'])


# List image files
image_files = os.listdir(samples_path)
image_files.sort()

# Display learning rate data
st.header("Learning Rates")
for column in lr_data.columns:
    if column != "Iteration":
        st.subheader(column)
        st.line_chart(lr_data[['Iteration', column]].set_index('Iteration'), height=300)

# Display inception score data
st.header('Inception Score')
st.line_chart(is_data.set_index('Iteration'), height=300)

# Display metrics data
st.header('Loss Metrics')
for column in metrics_data.columns:
    if column != 'Iteration':
        st.subheader(column)
        st.line_chart(metrics_data[['Iteration', column]].set_index('Iteration'), height=300)

# Display discriminator gradients data
st.header('Discriminator Gradients')
discriminator_data = gradients_data.loc[gradients_data["Component"] == "discriminator"].drop(columns=["Component"])
for column in discriminator_data.columns:
    if column != 'Iteration':
        st.subheader(column)
        st.line_chart(discriminator_data[['Iteration', column]].set_index('Iteration'), height=300)

# Display generator gradients data
st.header('Generator Gradients')
generator_data = gradients_data.loc[gradients_data["Component"] == "generator"].drop(columns=["Component"])
for column in generator_data.columns:
    if column != 'Iteration':
        st.subheader(column)
        st.line_chart(generator_data[['Iteration', column]].set_index('Iteration'), height=300)


# Display images
st.header('Sample Outputs')
selected_images = st.multiselect('Select Images', image_files)  # Allows multiple selections
for image in selected_images:
    st.image(os.path.join(samples_path, image), caption=image, use_column_width='auto')  # Display each selected image
