from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification


def create_classifier():
    model_name = "typeform/distilbert-base-uncased-mnli" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline('zero-shot-classification',
                          model=model,
                          tokenizer=tokenizer)
    return classifier


def predict(text, labels):
    classifier = create_classifier()
    candidate_labels = labels
    result = classifier(text, candidate_labels)
    return result['labels'][0]
