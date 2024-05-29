import functions

text = input()
candidate_labels = ["business", "sports", "politics", "technology"]

result = functions.predict(text, candidate_labels)

print(result[0][0])
