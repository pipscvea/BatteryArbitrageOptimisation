import train_data_assembly
import strategy
import train_func
from joblib import dump

from sklearn.model_selection import train_test_split

X = strategy.X
y = strategy.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = train_func.train(X_train, y_train, X_test, y_test)






