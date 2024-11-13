import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


def get_data():
    # загружаем положительные твиты
    positive = pd.read_csv('./task_1/data/positive.csv', sep=';',
                           usecols=[3], names=['text'])
    positive['label'] = ['positive'] * len(positive)  # устанавливаем метки

    # загружаем отрицательные твиты
    negative = pd.read_csv('./task_1/data/negative.csv', sep=';',
                           usecols=[3], names=['text'])
    negative['label'] = ['negative'] * len(negative)  # устанавливаем метки

    # соединяем вместе
    df = pd.concat([positive, negative])

    x_train, x_test, y_train, y_test = train_test_split(df.text, df.label)

    return [[x_train, y_train], [x_test, y_test]]
