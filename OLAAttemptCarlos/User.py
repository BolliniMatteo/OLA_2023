"""
    User class, with features defining its type
"""
from configs.configuration import class_config, feature_space


class User:

    def __init__(self, features=None, class_name=None):
        """
        Constructor function

        :param features: binary features determining class of User
        :param class_name: class name
        """

        self.class_name = class_name
        if features is None and class_name is not None:
            self.features = class_config[self.class_name]

    def map_feature_value_to_string(self):
        """
        Returns value of binary features

        :return: features in string format
        """
        feature_string_format = []
        for i, j in enumerate(self.features):
            feature_value = feature_space[list(feature_space.keys())[i]][j]
            feature_string_format.append(feature_value)
        return feature_string_format
