from transformers import AutoTokenizer


sentence = "Hello world!"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

print("\nToken Details:")
for token_id in token_ids:
    token = tokenizer.convert_ids_to_tokens(token_id)
    print(f"Token ID: {token_id}, Token: {token}")

print("\nDecoded Tokens:")
for token_id in token_ids:
    print(f"Token ID: {token_id}, Token: {tokenizer.decode([token_id])}")


# A list of colors in RGB for representing the tokens
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer_name: str):
    """ Show the tokens each separated by a different color """

    # Load the tokenizer and tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length
    print(f"Vocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )
print("\n\n")  # Add a newline after printing all tokens
print("Tokenization for different tokenizers: BERT Base Cased, BERT Base Uncased, and Xenova GPT-4\n")
print("BERT Base Cased:")
show_tokens(sentence, "bert-base-cased")
print("\n\nBERT Base Uncased:")
show_tokens(sentence, "bert-base-uncased")
print("\n\nXenova GPT-4:")
show_tokens(sentence, "Xenova/gpt-4")