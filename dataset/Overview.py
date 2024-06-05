#%%
import pandas as pd
import os 
import glob

#%%

base_dir = os.getcwd()

# Check data numbers in different categories

#Specify the certain file path
'''
file_paths = [
    './Picturess/e_close/motion/trial_0.csv',
    './Pictures/e_close/noun/trial_0.csv',
    './Pictures/e_open/motion/trial_0.csv',
    './Pictures/e_open/noun/trial_0.csv',
    './Imagination/e_close/trial_0.csv',
]
'''
#print(base_dir)
#file_paths = glob.glob(base_dir + '/**/*.csv', recursive=True)
file_paths = glob.glob(base_dir + '/without_individuals/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    counts = df.iloc[:, 0].value_counts()
    print(file_path + ":")
    print(counts)
    print()

    # Check if it don't have any vacancy in the data field
    if df.isnull().values.any():
        print("Ohh no!! There are some NaN in the data.")
    else:
        print("There are no NaN in the data.")
    print()
#%% 
#Modify the data in the CSV file
df_trail_9 = pd.read_csv('./Pictures/e_open/motion/trial_9.csv')
#cat = df_trail_9.loc[df_trail_9['Image'] == 'Cat']

#Replace the "Image" column from "Cat" to "Catch"
df_trail_9['Image'] = df_trail_9['Image'].replace('Cat', 'Catch')

#Save the modified data to the CSV file
df_trail_9.to_csv('./Pictures/e_open/motion/trial_9.csv', index=False)

#%%
#Check if the data got modified
df_trail_9 = pd.read_csv('./Pictures/e_open/motion/trial_9.csv')
catch = df_trail_9.loc[df_trail_9['Image'] == 'Catch']
print(catch)

#%%
#Change the index name of the CSV file
base_dir = os.getcwd()
file_paths = glob.glob(base_dir + '/**/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    #Check if any column is named as "TF9"
    if any("TF9" in col for col in df.columns):
        df.rename(columns=lambda x: x.replace('TF9', 'TP9'), inplace=True)
        df.to_csv(file_path, index=False)
        
#%%
#Combine the data in the CSV file with the same categories without differentiating the individual trials
#and save it to the new CSV file in the without_individual folder
def merge(csv_files):
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list).sort_values('Image')
    return merged_df

csv_files = glob.glob('./Pictures/e_close/motion/trial_*.csv')
merged_df = merge(csv_files)
merged_df.to_csv('./without_individuals/e_close/pic_e_close_motion.csv', index=False)

csv_files = glob.glob('./Pictures/e_close/noun/trial_*.csv')
merged_df = merge(csv_files)
merged_df.to_csv('./without_individuals/e_close/pic_e_close_noun.csv', index=False)

csv_files = glob.glob('./Pictures/e_open/motion/trial_*.csv')
merged_df = merge(csv_files)
merged_df.to_csv('./without_individuals/e_open/pic_e_open_motion.csv', index=False)

csv_files = glob.glob('./Pictures/e_open/noun/trial_*.csv')
merged_df = merge(csv_files)
merged_df.to_csv('./without_individuals/e_open/pic_e_open_noun.csv', index=False)

csv_files = glob.glob('./Imagination/e_close/trial_*.csv')
merged_df = merge(csv_files)
merged_df.to_csv('./without_individuals/imagination.csv', index=False)
