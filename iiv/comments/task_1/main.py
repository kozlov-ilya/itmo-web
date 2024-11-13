from utils.models import models
from utils.combine_reports import *
from utils.get_data import *
from utils.analyze_comments import *


data_train, data_test = get_data()

models_reports = []

for m in models:
    model = m['model']
    vectorizer = m['vectorizer']

    report = analyze_comments(data_train, data_test, model, vectorizer)

    models_reports.append([m['labels'], report])

    print(f"Report added {len(models_reports)}/{len(models)}")

result = combine_reports(models_reports)

df = pd.DataFrame(result)
df.to_csv('./task_1/comments_classification_reports.csv', index=False)
