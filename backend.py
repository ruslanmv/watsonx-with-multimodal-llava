import os
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import matplotlib.pyplot as plt
import urllib3
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
import textwrap
import pandas as pd
import numpy as np
from haversine import haversine  # Install haversine library: pip install haversine
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")


import os
import requests

url = 'https://github.com/ruslanmv/watsonx-with-multimodal-llava/raw/master/geocoded_hotels.csv'
filename = 'geocoded_hotels.csv'

# Check if the file already exists
if not os.path.isfile(filename):
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File {filename} downloaded successfully!")
    else:
        print(f"Error downloading file. Status code: {response.status_code}")
else:
    print(f"File {filename} already exists.")

import os
import pandas as pd
from datasets import load_dataset
import pyarrow

# 1. Get the Current Directory
current_directory = os.getcwd()

# 2. Construct the Full Path to the CSV File
csv_file_path = os.path.join(current_directory, 'hotel_multimodal.csv')

# 3. Check if the file exists
if not os.path.exists(csv_file_path):
    # If not, download the dataset
    print("File not found, downloading from Hugging Face...")

    dataset = load_dataset("ruslanmv/hotel-multimodal")

    # Convert the 'train' dataset to a DataFrame using .to_pandas()
    df_hotels = dataset['train'].to_pandas()

    # 4.Save to CSV
    df_hotels.to_csv(csv_file_path, index=False)  
    print("Dataset downloaded and saved as CSV.")


# 5. Read the CSV file
df_hotels = pd.read_csv(csv_file_path)

print("DataFrame loaded:")
geocoded_hotels_path = os.path.join(current_directory, 'geocoded_hotels.csv')
# Read the CSV file
geocoded_hotels = pd.read_csv(geocoded_hotels_path)

import requests

def get_current_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()

        location = data.get('loc', '')
        if location:
            latitude, longitude = map(float, location.split(','))
            return latitude, longitude
        else:
            return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

latitude, longitude = get_current_location()
if latitude and longitude:
    print(f"Current location: Latitude = {latitude}, Longitude = {longitude}")
else:
    print("Could not retrieve the current location.")


from geopy.geocoders import Nominatim

def get_coordinates(location_name):
    """Fetches latitude and longitude coordinates for a given location name.

    Args:
        location_name (str): The name of the location (e.g., "Rome, Italy").

    Returns:
        tuple: A tuple containing the latitude and longitude (float values),
               or None if the location is not found.
    """

    geolocator = Nominatim(user_agent="coordinate_finder")
    location = geolocator.geocode(location_name)

    if location:
        return location.latitude, location.longitude
    else:
        return None  # Location not found



def find_nearby(place=None):
    if place!=None:
        coordinates = get_coordinates(place)
        if coordinates:
            latitude, longitude = coordinates
            print(f"The coordinates of {place} are: Latitude: {latitude}, Longitude: {longitude}")
        else:
            print(f"Location not found: {place}")
    else:
        latitude, longitude = get_current_location()
        if latitude and longitude:
            print(f"Current location: Latitude = {latitude}, Longitude = {longitude}")
    # Load the geocoded_hotels DataFrame
    current_directory = os.getcwd()
    geocoded_hotels_path = os.path.join(current_directory, 'geocoded_hotels.csv')
    geocoded_hotels = pd.read_csv(geocoded_hotels_path)

    # Define input coordinates for the reference location
    reference_latitude = latitude
    reference_longitude = longitude

    # Haversine Distance Function
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        """Calculates the Haversine distance between two points on the Earth's surface."""
        return haversine((lat1, lon1), (lat2, lon2))

    # Calculate distances to all other points in the DataFrame
    geocoded_hotels['distance_km'] = geocoded_hotels.apply(
        lambda row: calculate_haversine_distance(
            reference_latitude, reference_longitude, row['latitude'], row['longitude']
        ),
        axis=1
    )

    # Sort by distance and get the top 5 closest points
    closest_hotels = geocoded_hotels.sort_values(by='distance_km').head(5)

    # Display the results
    print("The 5 closest locations are:\n")
    print(closest_hotels)
    return closest_hotels


def search_hotel(place=None):
    import os
    import pandas as pd
    import requests
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    import urllib3
    from transformers import pipeline
    from transformers import BitsAndBytesConfig
    import torch

    # Suppress the InsecureRequestWarning
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 1. Get the Current Directory
    current_directory = os.getcwd()
    # 2. Construct the Full Path to the CSV File
    csv_file_path = os.path.join(current_directory, 'hotel_multimodal.csv')
    # Read the CSV file
    df_hotels = pd.read_csv(csv_file_path)
    geocoded_hotels_path = os.path.join(current_directory, 'geocoded_hotels.csv')
    # Read the CSV file
    geocoded_hotels = pd.read_csv(geocoded_hotels_path)

    # Assuming find_nearby function is defined elsewhere
    df_found = find_nearby(place)

    # Converting df_found[["hotel_id"]].values to a list
    hotel_ids = df_found["hotel_id"].values.tolist()

    # Extracting rows from df_hotels where hotel_id is in the list hotel_ids
    filtered_df = df_hotels[df_hotels['hotel_id'].isin(hotel_ids)]

    # Ordering filtered_df by the order of hotel_ids
    filtered_df['hotel_id'] = pd.Categorical(filtered_df['hotel_id'], categories=hotel_ids, ordered=True)
    filtered_df = filtered_df.sort_values('hotel_id').reset_index(drop=True)

    # Define the quantization config and model ID
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "llava-hf/llava-1.5-7b-hf"

    # Initialize the pipeline
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

    # Group by hotel_id and take the first 2 image URLs for each hotel
    grouped_df = filtered_df.groupby('hotel_id', observed=True).head(2)

    # Create a new DataFrame for storing image descriptions
    description_data = []

    # Download and generate descriptions for the images
    for index, row in grouped_df.iterrows():
        hotel_id = row['hotel_id']
        hotel_name = row['hotel_name']
        image_url = row['image_url']

        try:
            response = requests.get(image_url, verify=False)
            response.raise_for_status()  # Check for request errors
            img = Image.open(BytesIO(response.content))

            # Generate description for the image
            prompt = "USER: <image>\nAnalyze this image and describe in detail what it contains. Provide a summary of the main elements and activities shown in the image. Additionally, give me feedback on whether this place is worth visiting and any precautions or items I should consider bringing with me. Provide a comprehensive review.\nASSISTANT:"
            outputs = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
            description = outputs[0]["generated_text"].split("\nASSISTANT:")[-1].strip()

            # Append data to the list
            description_data.append({
                'hotel_name': hotel_name,
                'hotel_id': hotel_id,
                'image': img,
                'description': description
            })
        except (requests.RequestException, UnidentifiedImageError):
            print(f"Skipping image at URL: {image_url}")

    # Create a DataFrame from the description data
    description_df = pd.DataFrame(description_data)
    return description_df


def show_hotels(place=None):
    description_df = search_hotel(place)

    # Calculate the number of rows needed
    num_images = len(description_df)
    num_rows = (num_images + 1) // 2  # Two images per row

    fig, axs = plt.subplots(num_rows * 2, 2, figsize=(20, 10 * num_rows))

    current_index = 0

    for _, row in description_df.iterrows():
        img = row['image']
        description = row['description']

        if img is None:  # Skip if the image is missing
            continue

        row_idx = (current_index // 2) * 2
        col_idx = current_index % 2

        # Plot the image
        axs[row_idx, col_idx].imshow(img)
        axs[row_idx, col_idx].axis('off')
        axs[row_idx, col_idx].set_title(f"{row['hotel_name']}\nHotel ID: {row['hotel_id']} Image {current_index + 1}", fontsize=16)

        # Wrap the description text
        wrapped_description = "\n".join(textwrap.wrap(description, width=50))

        # Plot the description
        axs[row_idx + 1, col_idx].text(0.5, 0.5, wrapped_description, ha='center', va='center', wrap=True, fontsize=14)
        axs[row_idx + 1, col_idx].axis('off')

        current_index += 1

    # Hide any unused subplots
    total_plots = (current_index + 1) // 2 * 2
    for j in range(current_index, total_plots * 2):
        row_idx = (j // 2) * 2
        col_idx = j % 2
        if row_idx < num_rows * 2:
            axs[row_idx, col_idx].axis('off')
        if row_idx + 1 < num_rows * 2:
            axs[row_idx + 1, col_idx].axis('off')

    plt.tight_layout()
    plt.show()

#place='Genova Italia'
#show_hotels(place)