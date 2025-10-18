import torch
import torch.nn as nn
from torch.nn import functional as F

# https://youtu.be/kCc8FmEb1nY?si=aGgfx2pTHzfJfTaM&t=3724

BATCH_SIZE = 32 # how many independent sequences will we process in parallel
# block size (context lenght). You don't want to train on the entire set at once. This is to compute intensive. So pick small blocks
BLOCK_SIZE = 8 # what is the maximum context lenght for prediction
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
NUMBER_OF_EMBEDDING_DIMENSIONS = 32

print(DEVICE)

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Get vocabulary size. Contains all possible characters in the dataset. The characters that can be emitted
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# One method to tokenize the vocabulary. It is a character based LLM so only translating individual characters into integers
# sub word tokenizer https://github.com/google/sentencepiece & https://github.com/openai/tiktoken
# You can make trade offs between the number of tokens and how many subword or characters they represent
character_to_integer_mapping = { character: index for index, character in enumerate(chars)}
integer_to_character_mapping = { index: character for index, character in enumerate(chars)}

encode = lambda text: [character_to_integer_mapping[character] for character in text]
decode = lambda numbers: ''.join([integer_to_character_mapping[integer] for integer in numbers])

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# Splitting up encoded dataset into training data and validation data. 
# If you take 100% the entire text is in memory and it will not creat new text
cuttof_value = int(0.9 * len(data))
train_data = data[:cuttof_value]
validation_data = data[cuttof_value:]

train_data[:BLOCK_SIZE + 1] # 9 characters gives 8 examples of training data. 
# It also gives the future context that model can use to predict text. This can be from one character input to 8 characters

# |Sampling: below the combination that create the examples are spelled out
x = train_data[:BLOCK_SIZE]
y = train_data[1:BLOCK_SIZE + 1]
for t in range(BLOCK_SIZE):
    context = x[:t+1]
    target = y[t]
    print(f'when input is {context} the target is: {target}')

print('-----------')

 # Now the above with a batch dimension. The GPU can run parallel processes so batching make efficient use of this capability
def get_batch(split):
    """
    Generate random batch data
    """
    data = train_data if split == 'train' else validation_data
    random_indexes = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[index : index + BLOCK_SIZE] for index in random_indexes])
    y = torch.stack([data[index + 1 : index + BLOCK_SIZE + 1] for index in random_indexes])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



input_arrays, target_arrays = get_batch('train')
print('inputs')
print(input_arrays.shape)
print(input_arrays)
print('targets')
print(target_arrays.shape)
print(target_arrays)

print('-----------')

for batch_index in range(BATCH_SIZE): # batch dimension
    for time_index in range(BLOCK_SIZE): # time dimension
        context = input_arrays[batch_index, : time_index + 1]
        target = target_arrays[batch_index, time_index]
        print(f'When input is {context.tolist()} the target: {target}')

print('-----------')

# Feeding the batches in to a neural network: most easy BigramLanguageModel
class BigramLanguageModel(nn.Module): # Base clas for all neural network modules
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits (predictions) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, NUMBER_OF_EMBEDDING_DIMENSIONS)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, NUMBER_OF_EMBEDDING_DIMENSIONS)
        self.language_modelling_head = nn.Linear(NUMBER_OF_EMBEDDING_DIMENSIONS, vocab_size)


    # idx is current context of some batch

    def forward(self, indices, targets=None): # The computation performed at every call
        B, T = indices.shape

        # indices and targets are both (B, T) tensor of integers
        token_embeddings = self.token_embedding_table(indices) # (B, T, C)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
        x = token_embeddings + positional_embeddings
        logits = self.language_modelling_head(x) # (B, T, vocab_size) -> (Batch (4), Time (8), Channel (65 or vocab size))


        # Conversion based on cross entropy doc
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets) # Are the predictions of logits any good? Quality test

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices of the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probalilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
model.to(DEVICE)
logits, loss = model(input_arrays, target_arrays)
print(logits.shape)
print(loss)
idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

batch_size = 32
for iter in range(MAX_ITERS):

    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["eval"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

