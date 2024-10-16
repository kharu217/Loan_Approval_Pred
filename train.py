from utils import *
import data
import model
import tqdm

model_t = model.loan_classify()

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model_t.parameters(), 1)

for i in range(epochs) :
    model_t.train()
    epoch_loss = 0
    for f, l in tqdm.tqdm(data.train_dataload) :
        optimizer.zero_grad()
        pred = model_t(torch.from_numpy(std_scale.fit_transform(f)))
        loss = loss_fn(pred.to(torch.float32), l)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'loss : {epoch_loss / len(data.train_dataload)}')
