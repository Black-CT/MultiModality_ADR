from transformers import AutoTokenizer

model_name = "gumgo91/IUPAC_BERT"  # Replace with the desired model name
tokenizer1 = AutoTokenizer.from_pretrained(model_name)

model_name2 = "superspider2023/iupacGPT"
tokenizer2 = AutoTokenizer.from_pretrained(model_name)


IUPAC1 = "N-(4-hydroxyphenyl)ethanmide"
IUPAC2 = "(2S)-1-[(2S)-2-[[(2S)-1-ethoxy-1-oxo-4-phenylbutan-2-yl]amino]-6-[(2,2,2-trifluoroacetyl)amino]hexanoyl]pyrrolidine-2-carboxylic acid"
IUPAC3 = "1-methyl-1-nitroso-3-[(3R,4R,5S,6R)-2,4,5-trihydroxy-6-(hydroxymethyl)oxan-3-yl]urea"

print(tokenizer1.tokenize(IUPAC1))
print(tokenizer2.tokenize(IUPAC1))

print(tokenizer1.tokenize(IUPAC2))
print(tokenizer2.tokenize(IUPAC2))

print(tokenizer1.tokenize(IUPAC3))
print(tokenizer2.tokenize(IUPAC3))