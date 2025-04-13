import pickle

# Open the pickle file in read-binary mode
with open('data/wigner_cat_minus.pickle', 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)
with open('wigner_cat.txt', 'w') as file:
    # Write the data to a text file
    for item in data:
        file.write(f"{item}\n")
# Now you can work with the loaded data
print(data)