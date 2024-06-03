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
    './picture/e_close/motion/trial_0.csv',
    './picture/e_close/noun/trial_0.csv',
    './picture/e_open/motion/trial_0.csv',
    './picture/e_open/noun/trial_0.csv',
    './imagination/e_close/trial_0.csv',
]
'''

file_paths = glob.glob(base_dir + '/**/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    counts = df.iloc[:, 0].value_counts()
    print(file_path + ":")
    print(counts)
    print()

    # Check if it don't have any vacancy in the data field
    if df.isnull().values.any():
        print("There are some NaN in the data.")
    else:
        print("There are no NaN in the data.")
    print()
#%% 
#Modify the data in the CSV file
df_trail_9 = pd.read_csv('./picture/e_open/motion/trial_9.csv')
#cat = df_trail_9.loc[df_trail_9['Image'] == 'Cat']

#Replace the "Image" column from "Cat" to "Catch"
df_trail_9['Image'] = df_trail_9['Image'].replace('Cat', 'Catch')

#Save the modified data to the CSV file
df_trail_9.to_csv('./picture/e_open/motion/trial_9.csv', index=False)

#%%
#Check if the data got modified
df_trail_9 = pd.read_csv('./picture/e_open/motion/trial_9.csv')
catch = df_trail_9.loc[df_trail_9['Image'] == 'Catch']
print(catch)