import torch
from sklearn.model_selection import train_test_split

Y = torch.stack(
    [G.y.squeeze(0) for G in GraphList]
)

Mean = Y.mean(dim=0)
Std = Y.std(dim=0)
print("Target Means")
print(Mean)
print()
print("Target Stds")
print(Std)

for G in GraphList:
    G.y = ((G.y.squeeze(0) - Mean)/Std).unsqueeze(0)

Indices = list(range(len(GraphList)))
TrainIdx, TempIdx = train_test_split(Indices,test_size=0.20,random_state=42)

ValIdx, TestIdx = train_test_split(
    TempIdx,
    test_size=0.50,
    random_state=42
)

print()
print("Train:", len(TrainIdx))
print("Val:", len(ValIdx))
print("Test:", len(TestIdx))

torch.save(
    GraphList,
    "processed_graphs.pt"
)

torch.save(
    {
        "mean": Mean,
        "std": Std,
        "train_idx": TrainIdx,
        "val_idx": ValIdx,
        "test_idx": TestIdx
    },
    "dataset_info.pt"
)

print()
print("Saved:")
print("processed_graphs.pt")
print("dataset_info.pt")