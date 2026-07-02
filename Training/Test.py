import torch
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


Model.load_state_dict(torch.load("arcevl_zero.pt", map_location=Device))
Model.eval()

Preds, Targets = [], []

with torch.no_grad():
    for Batch in TestLoader:

        Batch = Batch.to(Device)
        Out = Model(Batch.x, Batch.edge_index, Batch.edge_attr, Batch.batch)

        Preds.append(Out.cpu())
        Targets.append(Batch.y.squeeze(1).cpu())

Preds = torch.cat(Preds)
Targets = torch.cat(Targets)
Preds = Preds*Std + Mean
Targets = Targets*Std + Mean
Preds = Preds.numpy()
Targets = Targets.numpy()

Names = ["Params", "FLOPs", "SynFlow", "NWOT", "ZEN"]
print("\nTest Results")
print("-" * 40)

for i, Name in enumerate(Names):
    R2 = r2_score(Targets[:, i], Preds[:, i])
    MAE = mean_absolute_error(Targets[:, i], Preds[:, i])
    RMSE = np.sqrt(mean_squared_error(Targets[:, i], Preds[:, i]))
    print(f"\n{Name}")
    print(f"R²   : {R2:.6f}")
    print(f"MAE  : {MAE:.6f}")
    print(f"RMSE : {RMSE:.6f}")