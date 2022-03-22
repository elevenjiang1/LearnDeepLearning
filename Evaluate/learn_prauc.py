"""
This python file is to learn PR curves and AUC score
"""
import os
import sys
import numpy as np

Abs_Path=os.path.dirname(os.path.abspath(__file__))

def sort_function(data):
    data=data[:-4]
    data=data.split("xample_data_")[1]
    return int(data)

def calculate_error_rate(data_type='aug'):
    all_data_path=os.path.join(Abs_Path,"data/{}".format(data_type))
    files_name=os.listdir(all_data_path)

    files_name=sorted(files_name,key=sort_function)
    print(files_name)
    
    #calculate error rate
    for file_name in files_name:
        print("***files name is {}******".format(file_name))
        file_path=os.path.join(all_data_path,file_name)
        all_prediction=np.load(file_path)['all_predictions']
        all_labels=np.load(file_path)['all_labels']

        all_prediction[all_prediction>0.5]=1
        all_prediction[all_prediction<=0.5]=0
        
        new_array=np.abs(all_prediction-all_labels)
        error_rate=np.sum(new_array)/all_prediction.shape[0]
        print("error rate is:{}".format(error_rate))


def load_data(data_type="aug"):
    all_data_path=os.path.join(Abs_Path,"data/{}".format(data_type))
    files_name=os.listdir(all_data_path)

    files_name=sorted(files_name,key=sort_function)
    print(files_name)
    
    #calculate error rate
    for file_name in files_name:
        print("***files name is {}******".format(file_name))
        file_path=os.path.join(all_data_path,file_name)
        all_prediction=np.load(file_path)['all_predictions']
        all_labels=np.load(file_path)['all_labels']

        print(all_prediction[:10])
        print(all_labels[:10])
        all_prediction[all_prediction>0.5]=1
        all_prediction[all_prediction<=0.5]=0
        

        new_array=np.abs(all_prediction-all_labels)
        print(new_array[:10])

        error_rate=np.sum(new_array)/all_prediction.shape[0]
        print("error rate is:{}".format(error_rate))


        # sys.exit()
    


if __name__ == '__main__':
    load_data()