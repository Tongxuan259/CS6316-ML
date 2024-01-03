import os
import numpy as np

def load_data(data_path):
    data_file = open(data_path)
    data_lines = data_file.readlines()
    data_feature, data_target = [], []
    for line in data_lines:
        sample = line.strip('\n').split('\t')
        
        if "Present" in sample or "Absent" in sample:
            
            str_idx = sample.index("Present") if "Present" in sample else sample.index("Absent")
            sample[str_idx] = 1 if "Present" in sample else -1
        sample = [float(sample[i]) for i in range(len(sample))]
        data_feature.append(sample[0: -1])
        data_target.append(sample[-1])
    
    return {
        "data": np.array(data_feature),
        "target": np.array(data_target),
        "target_names": np.array([0, 1])
    }


if __name__ == "__main__":
    # data = load_data(data_path="./project3_dataset2.txt")
    # print(data)
    # print(data["data"][1])
    # print(data["target"].shape)
    pass
