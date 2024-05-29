from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
from typing import List, Tuple, Union


def initialize_classifier(model_name: str):
    """
    Initialize and return a zero-shot classification pipeline.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        pipeline: The initialized zero-shot classification pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline('zero-shot-classification',
                          model=model,
                          tokenizer=tokenizer)
    return classifier


def predict(texts: Union[str, List[str]],
            labels: List[str]) -> List[Tuple[str, float]]:
    """
    Predict the most likely label for each
    text from a list of candidate labels.

    Args:
        texts (Union[str, List[str]]): The input text or a list of texts.
        labels (List[str]): A list of candidate labels.

    Returns:
        List[str]: A list of the predicted labels.
    """
    if not isinstance(texts, (list, str)):
        raise ValueError("The 'texts' parameter should be" +
                         " a string or a list of strings.")
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(labels, list) or not all(isinstance(label, str)
                                               for label in labels):
        raise ValueError("The 'labels' parameter should be a list of strings.")

    if not texts:
        raise ValueError("The 'texts' list should not be empty.")
    if not labels:
        raise ValueError("The 'labels' list should not be empty.")

    if any(not text.strip() for text in texts):
        raise ValueError("The 'texts' list should not contain empty strings.")
    if any(not label.strip() for label in labels):
        raise ValueError("The 'labels' list should not contain empty strings.")

    classifier = initialize_classifier("typeform/distilbert-base-uncased-mnli")
    results = classifier(texts, labels)

    return [result['labels'][0] for result in results]
