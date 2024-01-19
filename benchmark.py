import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

path_to_data = ""
# Load files
x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)
y_train = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)

x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)

# Encoding
le = LabelEncoder()
x_train["type_territoire"] = le.fit_transform(x_train["type_territoire"])

# Missing values
x_train.fillna(x_train.mean(), inplace=True)


# Fit model
model = RandomForestRegressor(n_estimators = 10, random_state=0, max_depth=5, criterion="absolute_error")
model.fit(x_train, y_train.values.ravel())

# Prediction
x_test["type_territoire"] = le.fit_transform(x_test["type_territoire"])
x_test.fillna(x_test.mean(), inplace=True)
predicted_values = model.predict(x_test)

y_pred = pd.DataFrame(data=predicted_values, index=x_test.index.values, columns=["pertes_totales"])
y_pred.index.name = 'ID'

y_pred.to_csv(path_to_data + "/testing_benchmark.csv")
