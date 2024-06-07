#%%
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

#%%
file_paths = glob.glob('../dataset/without_individuals/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    plt.hist(df['Image'], bins=30, alpha=0.5)
    plt.title('Distribution of "Image" in ' + file_path)
    plt.xlabel('Image')
    plt.ylabel('The numbers of Image')
    plt.show()

#%%

#Check the average values for different channels of the data by separating the data by "Image"
file_paths = glob.glob('../dataset/without_individuals/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df['Image'] = df['Image'].astype('category')
    image_data = df.groupby("Image", observed=True).mean().round(2)
    print(file_path)

    csv_name = os.path.splitext(os.path.basename(file_path))[0]
    image_data = image_data.transpose()
    for column in image_data.columns:
        plt.scatter(image_data.index, image_data[column], label=column)

    plt.title('Average values for ' + csv_name)
    plt.xticks(rotation=45)
    plt.xlabel('Different Waves of categories and sensors')
    plt.ylabel('Average value')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
#%%
#Check the average and the standard deviation values for different channels of the data by separating the data by "Image"
for file_path in file_paths:
    df = pd.read_csv(file_path)
    df['Image'] = df['Image'].astype('category')
    
    image_mean = df.groupby("Image", observed=True).mean().round(2)
    image_std = df.groupby("Image", observed=True).std().round(2)
    
    csv_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for column in image_mean.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(image_mean.index, image_mean[column], label='Mean')
        plt.scatter(image_std.index, image_std[column], label='Std')
        plt.title('Mean and Std values for ' + column + ' in ' + csv_name)
        plt.xticks(rotation=45)
        plt.xlabel('Image')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.violinplot(x=df['Image'], y=df[column])
        plt.title('Violin plot for ' + column + ' in ' + csv_name)
        plt.xticks(rotation=45)
        plt.show()