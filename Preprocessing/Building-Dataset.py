import pandas as pd
from tqdm import tqdm

Rows = []

for ArchStr, ArchData in tqdm(NB201.items()):
    ImgNet = ArchData["ImageNet16-120"]
    ArchTuple = str(
        convert_str_to_op_indices(
            ArchStr
        )
    )
    SynFlow = SynFlowMap.get(
        ArchTuple,
        None
    )
    NWOT = NWOTMap.get(
        ArchTuple,
        None
    )
    ZEN = ZENMap.get(
        ArchTuple,
        None
    )
    Rows.append(
        {
            "arch_str": ArchStr,
            "arch_tuple": ArchTuple,
            "params": ImgNet["params"],
            "flops": ImgNet["flop"],
            "synflow": SynFlow,
            "nwot": NWOT,
            "zen": ZEN
        }
    )

MasterDf = pd.DataFrame(Rows)
print("Before NaN Filtering:", len(MasterDf))
MasterDf = MasterDf.dropna()
print("After NaN Filtering:", len(MasterDf))
print()
print(MasterDf.head())
MasterDf.to_csv(
    "master_labels.csv",
    index=False
)

print()
print("Saved:")
print("master_labels.csv")

MasterDf = MasterDf[
    (MasterDf["nwot"] > -1e8)
    &
    (MasterDf["zen"] > -1e8)
]

MasterDf = MasterDf.reset_index(drop=True)
print(len(MasterDf))
print(
    MasterDf[
        ["params","flops","synflow","nwot","zen"]
    ].describe()
)