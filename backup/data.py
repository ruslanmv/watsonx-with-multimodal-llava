import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import os

# File Path:
current_directory = os.getcwd()  # Get the current working directory
csv_file_path = os.path.join(current_directory, 'hotel_info.csv')  # Construct the full path to the CSV file

# Read the Data:
df_hotels = pd.read_csv(csv_file_path)#.head(100)  # Read the CSV file into a pandas DataFrame

# Geocoding Setup:
geolocator = Nominatim(user_agent="hotels")  # Create a Nominatim geolocator object (set a user agent)
output_file = "geocoded_hotels.csv"      # Define the output CSV file name

# Reverse Geocoding Function:
def reverse_geocode(lat, lon):
    try:
        # Geocoding Request:
        location = geolocator.reverse((lat, lon), exactly_one=True)  # Get address details for the given latitude and longitude

        # Extract Address Information:
        if location and location.raw and 'address' in location.raw:
            address = location.raw['address']
            return {
                'city': address.get('city', ''),         # Extract city name
                'country': address.get('country', ''),   # Extract country name
                'state': address.get('state', ''),       # Extract state name
                'county': address.get('county', ''),     # Extract county name (if available)
                'suburb': address.get('suburb', ''),     # Extract suburb name (if available)
                'postcode': address.get('postcode', ''), # Extract postcode
                'road': address.get('road', ''),         # Extract road/street name
                'house_number': address.get('house_number', '')  # Extract house number (if available)
            }
        else:
            return {}  # Return empty dictionary if address details are not found
    except GeocoderTimedOut:
        print("GeocoderTimedOut: Retrying...")         # Print error message if a timeout occurs
        time.sleep(5)  # Wait for 5 seconds before retrying
        return reverse_geocode(lat, lon)  # Retry the geocoding request

# Batch Processing:
batch_size = 500       # Number of rows to process in each batch
num_batches = len(df_hotels) // batch_size + 1  # Calculate the total number of batches needed
pause_duration = 60    # Pause duration between batches (in seconds)

first_batch = True     # Flag to track if it's the first batch

for batch_num in range(num_batches):
    # Determine Batch Indices:
    start_index = batch_num * batch_size       
    end_index = min((batch_num + 1) * batch_size, len(df_hotels))  # Ensure the last batch doesn't exceed the dataframe length

    batch_df = df_hotels.iloc[start_index:end_index]  # Get the current batch of rows
    
    # Process Batch:
    for index, row in batch_df.iterrows():
        latitude = row['latitude']   # Get latitude from the row
        longitude = row['longitude'] # Get longitude from the row

        address_info = reverse_geocode(latitude, longitude)  # Get address details

        # Update DataFrame:
        for key, value in address_info.items():  # Update the DataFrame with the retrieved address details
            df_hotels.at[index, key] = value  

        time.sleep(1)  # Sleep for 1 second to avoid hitting rate limits

    # Save Batch Results:
    if first_batch:   # If it's the first batch, write with headers
        df_hotels.iloc[start_index:end_index].to_csv(output_file, index=False)  
        first_batch = False
    else:             # Append subsequent batches without headers
        df_hotels.iloc[start_index:end_index].to_csv(output_file, mode='a', header=False, index=False)

    # Pause Between Batches (except for the last batch):
    if batch_num < num_batches - 1:  
        print(f"Batch {batch_num + 1} completed. Pausing for {pause_duration} seconds...")
        time.sleep(pause_duration)
