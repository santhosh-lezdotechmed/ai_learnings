import requests
import pandas as pd

# URL of the dataset
url = "https://huggingface.co/datasets/AtulManjhi/Medical-helthcare/resolve/main/medical_meadow_mediqa2.csv"

# Send a GET request to fetch the dataset
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Save the content to a CSV file
    with open("medical_meadow_mediqa2.csv", "wb") as f:
        f.write(response.content)
    print("Dataset successfully downloaded as 'medical_meadow_mediqa2.csv'.")
else:
    print(f"Failed to download the dataset. Status code: {response.status_code}")


import pandas as pd

# Load the dataset
file_path = "C:/Users/Santhosh.M/Documents/medical_meadow_mediqa.csv"  # Update this path to the location of your CSV file
df = pd.read_csv(file_path)

# Replace 'query' and 'answer' with the actual column names from the dataset
queries = df['input'].tolist()
answers = df['output'].tolist()

# Save to a text document
with open("queries_and_answers.txt", "w", encoding="utf-8") as file:
    for i, (query, answer) in enumerate(zip(queries, answers), 1):
        file.write(f"{i}. Query: {query}\n   Answer: {answer}\n\n")

print("Queries and answers have been saved to 'queries_and_answers.txt'.")
