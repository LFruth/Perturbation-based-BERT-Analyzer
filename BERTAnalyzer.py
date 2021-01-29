from transformers import BertModel, BertTokenizer
from Interpreter import Interpreter
import torch
import json
import matplotlib.pyplot as plt
import numpy as np


# Use CUDA (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the text you want to visualize
text = "This is a sentence to visualize the importance of each word in the intermediate layers of BERT."

# Load the tokenizer:
# You can use any base BERT model here
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]

# Load the model:
# You can use any base BERT model here
model = BertModel.from_pretrained("bert-base-cased").to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

# Get the input (x) from the words using the tokenizer
tokenized_ids = tokenizer.convert_tokens_to_ids(words)
segment_ids = [0 for _ in range(len(words))]
token_tensor = torch.tensor([tokenized_ids], device=device)
segment_tensor = torch.tensor([segment_ids], device=device)
x = model.embeddings(token_tensor, segment_tensor)[0]

# here, we load the pre-calculated regurlarization
regularization = json.load(open("regular.json", "r"))

# extract the Phi we need to explain
def Phi(x):
    global model
    x = x.unsqueeze(0)
    attention_mask = torch.ones(x.shape[:2]).to(x.device)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    model_list = model.encoder.layer[:layer]
    hidden_states = x
    for layer_module in model_list:
        if type(hidden_states) is tuple:  # Fix for the huggingface pre-trained models
            hidden_states = hidden_states[0]
        hidden_states = layer_module(hidden_states, extended_attention_mask)
    return hidden_states[0]

# set the layers you want to visualize
visualize_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# set the iterations
iterations = 2000
# array to store the sigma of each layer
sigmas = []
for i in visualize_layers:
    layer = i
    # Load and optimize the interpreter
    interpreter = Interpreter(x=x, Phi=Phi, scale=0.55, regularization=regularization, words=words).to(
        device
    )
    interpreter.optimize(iteration=iterations, lr=0.01, show_progress=True)
    sigmas.append(interpreter.get_sigma())
    print("layer #%d done." % layer)

sigmas = np.array(sigmas)

# plot the result
_, ax = plt.subplots()
ax.imshow(sigmas, cmap='GnBu_r')
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words)
ax.set_yticks(range(len(sigmas)))
ax.set_yticklabels(map(str, visualize_layers))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
plt.tight_layout()
plt.show()
plt.savefig("result")
