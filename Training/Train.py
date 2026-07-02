import torch
import torch.nn as nn

from trainer import train

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", Device)

Model = Model.to(Device)

Criterion = nn.MSELoss()
Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-3, weight_decay=1e-5)

Epochs = 50

print(f"\nTrain Graphs: {len(TrainLoader.dataset)}")
print(f"Val Graphs: {len(ValLoader.dataset)}")
print(f"Test Graphs: {len(TestLoader.dataset)}\n")

TrainHistory, ValHistory, BestValLoss = train(
    model=Model,
    train_loader=TrainLoader,
    val_loader=ValLoader,
    criterion=Criterion,
    optimizer=Optimizer,
    device=Device,
    epochs=Epochs
)

torch.save(
    {"train_loss": TrainHistory, "val_loss": ValHistory},
    "training_history.pt"
)

print("\nTraining Complete")
print("Best Validation Loss:", BestValLoss)

print("\nSaved:")
print("arcevl_zero.pt")
print("training_history.pt")