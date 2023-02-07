'''
the python script reads the files from the dataset folder and process the audio putting spectral gating on 
each of the sample in order to remove the background noises
The file requires noisereduce library to work 
to install : pip install noisereduce
'''
import librosa as lb
import soundfile as sf
import noisereduce as nr # the library to apply spectral Gating 
import os

lst_of_dirs = ['a','b','c','d','e','f'];

for i in range(len(lst_of_dirs)):
    folder_path = f'../01_Dataset Exploration/training-{lst_of_dirs[i]}' #specify the folder path, where the datasets are kept (Source)
    list_of_files = os.listdir(folder_path);
    list_of_files = [sound_files for sound_files in list_of_files if '.wav' in sound_files];
    print(f'\n directory {lst_of_dirs[i]} Starts!!!!')
    for j in range(len(list_of_files)):
        soundtrack = list_of_files[j];
        file_path = rf'../01_Data Exploration/training-{lst_of_dirs[i]}/{soundtrack}'; #fpath of the 
        y , sr = lb.load(file_path);
        d = nr.reduce_noise(y=y, sr=sr);
        path_to_save = fr'../Dataset_ready/training-f/{soundtrack}'; # specify the folder path, where the datasets are to be stored after Processing (Destination)
        sf.write(path_to_save, d, sr , subtype='PCM_24');
        callback_msg = f'f0{i} file edited and moved';
        print(callback_msg);

print('done');