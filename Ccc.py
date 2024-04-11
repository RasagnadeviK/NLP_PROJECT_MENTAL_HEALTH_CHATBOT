import json
import pandas as pd

# Load JSON data from file
with open('intents.json', 'r') as json_file:
    data = json.load(json_file)

# Convert JSON data to DataFrame
df = pd.DataFrame(data['intents'])

# Save DataFrame to CSV file
df.to_csv('intents.csv', index=False)