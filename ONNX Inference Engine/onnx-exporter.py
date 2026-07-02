import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

from models.Architecture_MainModel import ZeroGraph


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = ZeroGraph()
Model.load_state_dict(
    torch.load(
        "arcevl_zero.pt",
        map_location=Device
    )
)

Model = Model.to(Device)
Model.eval()


class ONNXExporter(nn.Module):
    def __init__(self):
        super().__init__()
        HiddenDim = 128
        self.NodeProj = nn.Linear(1, HiddenDim)
        self.Conv1 = Model.encoder.conv1
        self.Conv2 = Model.encoder.conv2
        self.Conv3 = Model.encoder.conv3
        self.Conv4 = Model.encoder.conv4
        self.Head = Model.head

    def forward(self, x, edge_index, edge_attr):
        x = self.NodeProj(x)
        x = self.Conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.Conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.Conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.Conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = x.mean(dim=0, keepdim=True)
        return self.Head(x)


ONNXModel = ONNXExporter().to(Device)
ONNXModel.NodeProj.load_state_dict(
    Model.encoder.node_proj.state_dict()
)

ONNXModel.Head.load_state_dict(
    Model.head.state_dict()
)
ONNXModel.eval()
Sample = torch.load("processed_graphs.pt")[0]

x = Sample.x.to(Device)
edge_index = Sample.edge_index.to(Device)
edge_attr = Sample.edge_attr.to(Device)

torch.onnx.export(
    ONNXModel,
    (x, edge_index, edge_attr),
    "arcevl_zero.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=[
        "x",
        "edge_index",
        "edge_attr"
    ],
    output_names=[
        "predictions"
    ]
)

print("ONNX Export Successful")
ModelONNX = onnx.load("arcevl_zero.onnx")
onnx.checker.check_model(ModelONNX)
print("ONNX Model Valid")