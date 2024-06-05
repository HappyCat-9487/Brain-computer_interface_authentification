#%%
import pandas as pd
import matplotlib.pyplot as plt
import glob

#%%
file_paths = glob.glob('../dataset/without_individuals/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    plt.hist(df['Image'], bins=30, alpha=0.5)
    plt.title('Distribution of "Image" in ' + file_path)
    plt.xlabel('Image')
    plt.ylabel('Frequency')
    plt.show()

#%%
