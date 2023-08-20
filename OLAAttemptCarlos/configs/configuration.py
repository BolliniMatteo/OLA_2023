"""
    Configuration file defining the features and the
    classes for each user
"""

# Feature space for users
feature_space = {
    'age': ['< 25', '> 25'],
    'occupation': ['student', 'worker']
}

# User's classes (types of gamer)
class_config = {
    'casual': [1, 1],  # > 25, worker
    'hardcore': [0, 1],  # < 25, student
    'frequent': [0, 0]  # < 25, worker
}