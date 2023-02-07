import pandas as pd;
from sklearn.model_selection import train_test_split;

# importing the original datasets and combining them into one dataset
odsa = pd.read_csv('../Datasets/References/REFERENCE_A.csv')
odsb = pd.read_csv('../Datasets/References/REFERENCE_B.csv')
odsc = pd.read_csv('../Datasets/References/REFERENCE_C.csv')
odsd = pd.read_csv('../Datasets/References/REFERENCE_D.csv')
odse = pd.read_csv('../Datasets/References/REFERENCE_E.csv')
odsf = pd.read_csv('../Datasets/References/REFERENCE_F.csv')

lst_of_datasets = [odsa , odsb, odsc, odsd, odse, odsf]
combined_dataset_of_original_dataset = pd.concat(lst_of_datasets , axis = 0)

X_original = combined_dataset_of_original_dataset['Sample']
Y_original = combined_dataset_of_original_dataset['class']

x_train_o , x_test_o , y_train_o , y_test_o = train_test_split(X_original , Y_original , test_size = 0.3 , shuffle=True)

# importing the mfcc 44 dataset and chroma features 
mdsa = pd.read_csv(fr'../Datasets/Extracted_features_modifies_mfcc44/training-a.csv')
mdsb = pd.read_csv(fr'../Datasets/Extracted_features_modifies_mfcc44/training-b.csv')
mdsc = pd.read_csv(fr'../Datasets/Extracted_features_modifies_mfcc44/training-c.csv')
mdsd = pd.read_csv(fr'../Datasets/Extracted_features_modifies_mfcc44/training-d.csv')
mdse = pd.read_csv(fr'../Datasets/Extracted_features_modifies_mfcc44/training-e.csv')
mdsf = pd.read_csv(fr'../Datasets/Extracted_features_modifies_mfcc44/training-f.csv')
lst_of_datasets = [mdsa , mdsb, mdsc, mdsd, mdse, mdsf]
combined_mfcc_dataset = pd.concat(lst_of_datasets , axis = 0)

cdsa = pd.read_csv(fr'../Datasets/ExtractedFeaturesChroma/training-a_hp.csv')
cdsb = pd.read_csv(fr'../Datasets/ExtractedFeaturesChroma/training-b_hp.csv')
cdsc = pd.read_csv(fr'../Datasets/ExtractedFeaturesChroma/training-c_hp.csv')
cdsd = pd.read_csv(fr'../Datasets/ExtractedFeaturesChroma/training-d_hp.csv')
cdse = pd.read_csv(fr'../Datasets/ExtractedFeaturesChroma/training-e_hp.csv')
cdsf = pd.read_csv(fr'../Datasets/ExtractedFeaturesChroma/training-f_hp.csv')
lst_of_datasets = [cdsa , cdsb, cdsc, cdsd, cdse, cdsf]
combined_chroma_dataset = pd.concat(lst_of_datasets , axis = 0)

combined_mfcc_dataset = combined_mfcc_dataset.drop(['class'] , axis = 1)
final_dataset = pd.concat([combined_mfcc_dataset , combined_chroma_dataset] , axis = 1)

final_dataset.replace(-1 , 0 , inplace=True)

list_of_training_data = [i+'.wav' for i in x_train_o]
list_of_testing_data = [i for i in y_train_o]
list_of_testing_data_ans = [i for i in y_test_o]
list_of_training_data_ans = [i+'.wav' for i in x_test_o]

X_train = final_dataset.loc[final_dataset['Sample'].isin(list_of_training_data)]
y_train = final_dataset.loc[final_dataset['Sample'].isin(list_of_training_data_ans)]

X_train.to_csv('training.csv', index = False)
y_train.to_csv('testing.csv', index = False)