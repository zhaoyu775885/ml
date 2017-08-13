import os
import numpy as np

def img2vec(file):
    fH = open(file, 'r')
    vec = []
#    count = 0
    for line in fH:
        for char in line:
            try:
                number = int(char)
                vec.append(number)
#                print(int(char))
#                count += 1
            except ValueError:
                continue
#    print(count)
    vec = np.array(vec)
#    print(vec)
    return vec

def dist(vec1, vec2):
    vec0 = vec1 - vec2
    return np.sqrt(np.dot(vec0, vec0))
    
def count_kNN(labels):
    digits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for label in labels[:]:
        digits[label] += 1

#    print(digits)
    for i in range(10):
        digits[i] = (i, digits[i])
    digits = sorted(digits, key=lambda id:id[1], reverse=True)
    return digits[0][0]
    
    
def kNN(dist, k):
    index_set = []
    feature_set = []
    feature_size = len(dist)
    for item in range(feature_size):
        feature_set.append((dist[item], item))
        
    feature_set = sorted(feature_set, key=lambda feature: feature[0], reverse=False)
        
    for it in range(k):
        index_set.append(feature_set[it][1])
        
    return index_set

def classify_kNN(sample, vec, k):
    distance = []
    for item in sample[:]:
        distance.append( dist(item[0], vec) )
        
    idx_set = kNN(distance, k)
    label_set = []
    for idx in idx_set[:]:
        label_set.append(sample[idx][1])
#    print(label_set)
    return count_kNN(label_set)
    
    
        
def testing(rootdir, sample, k):
    
    total = 0
    right = 0
    
    for item in os.listdir(rootdir):
        path = os.path.join(rootdir, item)
#        print(path)
        if os.path.isfile(path):
            total += 1
            string_list = item.split('_')
            true_label = int(string_list[0])
            vec = img2vec(path)
            test_label = classify_kNN(sample, vec, k)
#            print('True Label: {0}, Test Label: {1}'.format(true_label, test_label))
            if true_label == test_label:
                right += 1
#            break
        if os.path.isdir(path):
            tvs_dir(path)
    print(total, right, right/total)

def training(rootdir):
    feature = []
    for item in os.listdir(rootdir):
        path = os.path.join(rootdir, item)
#        print(path)
        if os.path.isfile(path):
            string_list = item.split('_')
            label = int(string_list[0])
            vec = img2vec(path)
            feature.append((vec, label))
#            break
        if os.path.isdir(path):
            tvs_dir(path)
    return feature

if __name__ == '__main__':
    training_data = 'digits/trainingDigits'
    test_data = 'digits/testDigits'
    k = 10
    
    print('begin training')
    feature = training(training_data)
    testing(test_data, feature, k)
