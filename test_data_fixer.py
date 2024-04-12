import pandas as pd

# Step 1: Read the CSV file
# Replace 'your_file.csv' with the path to your actual CSV file
df = pd.read_csv('./fashion_data/fasion-pairs-train.csv')

# Step 2: Extract Column 1 data (assuming the first column is named 'Column1', adjust as necessary)
column_1_data = df.iloc[:20000, 1]  # This selects all rows of the first column


def filter_csv_by_categories(input_csv_path, output_csv_path, categories_set):
    # Step 1: Read the second CSV file
    df = pd.read_csv(input_csv_path)

    # Step 2: Filter rows where the category is in the set
    # Assuming the column to check is the first one, adjust as necessary
    def is_category_in_set(name):
        category_info = extract_category(name)
        return category_info in categories_set

    filtered_df = df[df.iloc[:, 1].apply(is_category_in_set)]

    # Step 3: Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)


# Function to extract categories from the name
def extract_category(name):
    gender_category_part = name.split('fashion')[1]
    if "WOMEN" in gender_category_part:
        category = gender_category_part.split("WOMEN")[1].split("id")[0]
        return ("WOMEN_"+ category)
    elif "MEN" in gender_category_part:
        category = gender_category_part.split("MEN")[1].split("id")[0]
        return ("MEN_"+category)
    return (None)  # In case the pattern does not match

# Step 3: Process each name to find categories
categories_set = {extract_category(name) for name in column_1_data}

# Converting the set back to a list for any further processing or to print
categories_list = list(categories_set)

# Printing the extracted categories
for category in categories_list:
    if category:  # Checking if both gender and category were found
        print(f"Category: {category}")
    else:
        print("Pattern not found or does not match.")


input_csv_path = './fashion_data/fasion-pairs-test.csv'
output_csv_path = './fashion_data/fasion-pairs-test-filtered.csv'
filter_csv_by_categories(input_csv_path, output_csv_path, categories_list)