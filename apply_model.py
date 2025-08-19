from joblib import load
from training import X_test


loaded_model = load("refined_model1_2017_JantoMarch.joblib")

# test trained model to see how it would have performed on data