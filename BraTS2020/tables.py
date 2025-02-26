from prep import *
import pandas as pd
import seaborn as sns


data_path = r"D:\Datasets\BraTS2020_training_data\content\data"

meta_data_df = pd.read_csv(os.path.join(data_path, "meta_data.csv"))
meta_data_df

name_mapping_df = pd.read_csv(os.path.join(data_path, "name_mapping.csv"))
name_mapping_df

survival_info_df = pd.read_csv(os.path.join(data_path, "survival_info.csv"))
survival_info_df

plt.figure(figsize=(10, 6))
sns.histplot(survival_info_df['Age'], bins=50, kde=True)
plt.title('Распределение возраста пациентов')
plt.xlabel('Возраст')
plt.ylabel('Частота')
plt.grid(False)
plt.show()