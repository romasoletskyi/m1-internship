import numpy as np

def preprocessing_euclid(field):
    alpha = 0.5
    epsilon = 1e-6
    
    field_shifted = field - np.min(field) + epsilon
    field_powered = np.power(field_shifted, alpha)
    field_powered = (field_powered - field_powered.mean()) / field_powered.std()

    return fields_powered_shifted