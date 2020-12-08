import numpy as np

class InfoGains():
    def __init__(self, features: np.array, classes: np.array, class_types: tuple = (1,-1)):
        '''
        INPUT:
            features = numpy array of features (ex: Xtr)
            classes = numpy array of classes (ex: Ytr)
            class_types = tuple of class types (ex: (1,-1))
                (can only be BINARY)
        '''
        self.features = features
        self.classes = classes

        # currently, we only support binary class types
        assert len(class_types) == 2
        self.class_types = class_types

        # entropy of class data
        self.entropy = self.calc_entropy(self.classes)
        
        #print("entropy of class", self.entropy)

        # probabiltiies of class types based on class data
        # p_class_a refers to the probability of the first class type
        self.p_class_a = np.count_nonzero(self.classes == self.class_types[0]) / len(self.classes)

        # p_class_b refers to the probability of the second class th
        self.p_class_b = np.count_nonzero(self.classes == self.class_types[1]) / len(self.classes)


    def calc_entropy(self, classes: np.array) -> float:
        '''
        Calculate entropy of class data
        '''
        p_yes = np.count_nonzero(classes == self.class_types[0]) / len(classes)
        p_no = np.count_nonzero(classes == self.class_types[1]) / len(classes)

        #print(p_yes,p_no)
        entropy = - p_yes * np.log2(p_yes) - p_no * np.log2(p_no)

        return entropy


    def calc_entropy_of_feature(self, feature: np.array, feature_val: int) -> float:
        
        p_yes = np.count_nonzero(self.classes[feature == feature_val] == self.class_types[0]) / len(self.classes[feature == feature_val] == self.class_types[0])
        p_no = np.count_nonzero(self.classes[feature == feature_val] == self.class_types[1]) / len(self.classes[feature == feature_val] == self.class_types[1])

        # print(self.classes[feature == feature_val])
        if (p_yes == 0 or p_yes == 1):
            return 0

        entropy = - p_yes * np.log2(p_yes) - p_no * np.log2(p_no)

        # print('entropy:', entropy)
        # print(feature)
        # print(self.classes)


        return entropy


    def calc_info_gain_of_feature(self, feature: np.array) -> float:
        #print("--- info gain of feature ---")
        
        weighted_entropy = 0
        
        # get unique values and their corresponding occurences
        unique_values, occurences = np.unique(feature, return_counts=True)
        
        # zip them for convenient iteration     
        unique_values_occurences = zip(unique_values, occurences)

        for unique_value, occurences in unique_values_occurences:
            entropy_of_value = self.calc_entropy_of_feature(feature, unique_value)
            
            #print(f"occurences of value {unique_value} = {occurences}")
            #print(f"{len(self.classes[feature == unique_value])}")
            
            # calculate weighted entropy of this unique feature value
            # calculated by entropy * probability(that unique value)
            weighted_entropy_of_value = entropy_of_value * (occurences / len(feature))
            
            
            
            # add to overall weighed entropy
            weighted_entropy += weighted_entropy_of_value
            
        # calculate final information gain for this feature
        
        #print("weighted_entropy", weighted_entropy)
        info_gain = self.entropy - weighted_entropy
        
        #print("info gain", info_gain)

        return info_gain



    def calc_all_info_gain(self) -> np.array:
        gains = np.array(np.zeros(len(self.features.T)))

        for index,feature in enumerate(self.features.T):
            # print(f"calculating info gain for feature {index}")
            info_gain = self.calc_info_gain_of_feature(feature)

            gains[index] = info_gain

        return gains
