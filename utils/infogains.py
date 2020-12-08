import numpy as np

class InfoGains():
    def __init__(self, data: np.array, class_types: tuple = (0,1):
        '''
        Takes in data of features + classes
        also takes a tuple of class_types (must be binary)
        '''
        self.features = data[:,:-1]
        self.classes = data[:,-1]
                 
        assert len(class_types) == 2
        self.class_types = class_types

        self.entropy = self.calc_entropy(self.classes)
        self.p_class_a = np.count_nonzero(self.classes == 1) / len(self.classes)
        self.p_class_b = np.count_nonzero(self.classes == -1) / len(self.classes)


    def calc_entropy(self, classes: np.array) -> float:
        p_yes = np.count_nonzero(classes == 1) / len(classes)
        p_no = np.count_nonzero(classes == -1) / len(classes)

        entropy = - p_yes * np.log2(p_yes) - p_no * np.log2(p_no)

        return entropy


    def calc_entropy_of_feature(self, feature: np.array, feature_val: int) -> float:
        p_yes = np.count_nonzero(self.classes[feature == feature_val] == 1) / len(self.classes[feature == feature_val] == 1)
        p_no = np.count_nonzero(self.classes[feature == feature_val] == -1) / len(self.classes[feature == feature_val] == -1)

        # print(self.classes[feature == feature_val])
        if (p_yes == 0 or p_yes == 1):
            return 0

        entropy = - p_yes * np.log2(p_yes) - p_no * np.log2(p_no)

        print('entropy:', entropy)
        # print(feature)
        # print(self.classes)


        return entropy


    def calc_info_gain_of_feature(self, feature: np.array) -> float:
        #print("--- info gain of feature ---")
        entropy_left = self.calc_entropy_of_feature(feature,self.class_types[0])
        entropy_right = self.calc_entropy_of_feature(feature,self.class_types[1])


        weighted_entropy = entropy_left*(1 - feature.mean()) + entropy_right*(feature.mean())

        info_gain = self.entropy - weighted_entropy

        return info_gain



    def calc_all_info_gain(self) -> np.array:
        gains = np.array(np.zeros(len(self.features.T)))

        for index,feature in enumerate(self.features.T):
            info_gain = self.calc_info_gain_of_feature(feature)

            gains[index] = info_gain

        return gains
