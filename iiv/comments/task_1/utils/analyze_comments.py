from sklearn.metrics import *


def analyze_comments(train_data, test_data, model, vectorizer):
    x_train, y_train = train_data
    x_test, y_test = test_data

    vectorized_x_train = vectorizer.fit_transform(x_train)

    model.fit(vectorized_x_train, y_train)

    vectorized_x_test = vectorizer.transform(x_test)

    pred = model.predict(vectorized_x_test)

    return classification_report(y_test, pred, output_dict=True)
