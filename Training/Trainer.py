import torch
from tqdm import tqdm


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_val_loss = float("inf")
    train_history = []
    val_history = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()*batch.num_graphs

        train_loss = running_train_loss/len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(pred, batch.y.squeeze(1))

                running_val_loss += loss.item() * batch.num_graphs

        val_loss = running_val_loss/len(val_loader.dataset)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "arcevl_zero.pt")
            print("Best Model Saved")

    return train_history, val_history, best_val_loss