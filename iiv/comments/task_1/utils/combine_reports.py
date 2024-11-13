def combine_reports(models_reports):
    result = []

    DECIMAL_PLACES = 2

    for labels, report in models_reports:
        metrics = {
            'Model': labels["model"],
            'Vectorizer': labels["vectorizer"],
            'N-grams': labels["ngrams"],
            'Tokenizer': labels["tokenizer"] if "tokenizer" in labels else '_',
            'Noise': labels["noise"] if "noise" in labels else '_',
            'Analyzer': labels["analyzer"] if "analyzer" in labels else '_',

            'Accuracy': round(report['accuracy'], DECIMAL_PLACES),

            'Precision (w_avg)': round(report['weighted avg']['precision'], DECIMAL_PLACES),
            'Recall (w_avg)': round(report['weighted avg']['recall'], DECIMAL_PLACES),
            'F1-Score (w_avg)': round(report['weighted avg']['f1-score'], DECIMAL_PLACES),

            'Precision (m_agv)': round(report['macro avg']['precision'], DECIMAL_PLACES),
            'Recall (m_agv)': round(report['macro avg']['recall'], DECIMAL_PLACES),
            'F1-Score (m_agv)': round(report['macro avg']['f1-score'], DECIMAL_PLACES),

            'Precision (pos)': round(report['positive']['precision'], DECIMAL_PLACES) if 'positive' in report else None,
            'Recall (pos)': round(report['positive']['recall'], DECIMAL_PLACES) if 'positive' in report else None,
            'F1-Score (pos)': round(report['positive']['f1-score'], DECIMAL_PLACES) if 'positive' in report else None,

            'Precision (neg)': round(report['negative']['precision'], DECIMAL_PLACES) if 'negative' in report else None,
            'Recall (neg)': round(report['negative']['recall'], DECIMAL_PLACES) if 'negative' in report else None,
            'F1-Score (neg)': round(report['negative']['f1-score'], DECIMAL_PLACES) if 'negative' in report else None,
        }
        result.append(metrics)

    return result
