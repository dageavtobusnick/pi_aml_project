import unittest
from functions import predict

class TestPredict(unittest.TestCase):

    def test_predict_single_text(self):
        texts = "The weather is nice today."
        labels = ["weather", "news"]
        expected = ["weather"]
        result = predict(texts, labels)
        self.assertEqual(result, expected)

    def test_predict_multiple_texts(self):
        texts = ["The weather is nice today.", "I enjoy reading books."]
        labels = ["weather", "books"]
        expected = ["weather", "books"]
        result = predict(texts, labels)
        self.assertEqual(result, expected)
        
    def test_predict_multiple_news(self):
        texts = ["Economists predict strong economic growth for the upcoming fiscal year.",
                 "The results of the recent elections have been announced, with the incumbent party winning a majority.",
                 "Researchers have made significant advancements in artificial intelligence, leading to new breakthroughs.",
                 "The football club has signed a promising young player from the local academy."]
        labels = ["business", "sports", "politics", "technology"]
        expected = ["business", "politics", "technology", "sports"]
        result = predict(texts, labels)
        self.assertEqual(result, expected)

    def test_empty_texts(self):
        with self.assertRaises(ValueError):
            predict([], ["weather", "news"])

    def test_empty_labels(self):
        with self.assertRaises(ValueError):
            predict("The weather is nice today.", [])

    def test_invalid_input_types(self):
        with self.assertRaises(ValueError):
            predict(123, ["weather", "news"])

        with self.assertRaises(ValueError):
            predict("The weather is nice today.", "weather")

        with self.assertRaises(ValueError):
            predict(["The weather is nice today."], "weather")

if __name__ == '__main__':
    unittest.main()