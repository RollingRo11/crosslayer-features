from nnsight import LanguageModel

lm = LanguageModel("google/gemma-2-2b", device_map="auto")

print(lm)
