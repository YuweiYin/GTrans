input="35.2 & 21.3 & 39.6 & 15.3 & 18.8 & 20.6 & 26.3 & 14.8 & 17.1 & 12.8"
scores = input.strip().replace("  ", " ").replace("  ", " ").split('&')
scores = [float(s) for s in scores]
assert len(scores) == 10
print(sum(scores[:6])/6)
print(sum(scores[6:])/4)
print(sum(scores)/len(scores))