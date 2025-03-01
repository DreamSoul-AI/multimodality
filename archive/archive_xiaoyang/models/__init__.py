from .GPT_1 import *
from .lstm import *
def train(dataloader,FLAGS,model):
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  idx = 0
  for epoch in range(FLAGS.batch_iter):
    for X_batch in dataloader:
      train_loss, logits = model.full_loss(X_batch, with_grad=True)
      optimizer.step()
      optimizer.zero_grad()
      idx += 1
    if idx%FLAGS.print_iter == 0:
        print(f"{idx}th epoch", ":", train_loss.item())