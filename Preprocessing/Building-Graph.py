import torch
import pandas as pd

from torch_geometric.data import Data
from tqdm import tqdm

EDGE_LIST = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]

GraphList = []
for _, Row in tqdm(
    MasterDf.iterrows(),
    total=len(MasterDf)
):
    Ops = eval(
        Row["arch_tuple"]
    )
    EdgeAttr = []
    for Op in Ops:
        OneHot = [0.0] * 5
        OneHot[Op] = 1.0
        EdgeAttr.append(
            OneHot
        )
    EdgeAttr = torch.tensor(
        EdgeAttr,
        dtype=torch.float
    )
    EdgeIndex = torch.tensor(EDGE_LIST,dtype=torch.long).t().contiguous()
    X = torch.ones((4, 1), dtype=torch.float)
    Y = torch.tensor(
        [
            Row["params"],
            Row["flops"],
            Row["synflow"],
            Row["nwot"],
            Row["zen"]
        ],
        dtype=torch.float
    )
    Graph = Data(
        x=X,
        edge_index=EdgeIndex,
        edge_attr=EdgeAttr,
        y=Y.unsqueeze(0)
    )
    GraphList.append(
        Graph
    )

print()
print(
    "Graphs:",
    len(GraphList)
)
print()
print(
    GraphList[0]
)