import torch
import numpy as np
import onnxruntime as ort


Session = ort.InferenceSession(
    "arcevl_zero.onnx",
    providers=["CPUExecutionProvider"]
)

Sample = torch.load("processed_graphs.pt")[0]
x = Sample.x.to(Device)
edge_index = Sample.edge_index.to(Device)
edge_attr = Sample.edge_attr.to(Device)
Inputs = {
    "x": x.cpu().numpy().astype(np.float32),
    "edge_index": edge_index.cpu().numpy().astype(np.int64),
    "edge_attr": edge_attr.cpu().numpy().astype(np.float32)
}
ONNXOut = Session.run(
    None,
    Inputs
)[0]

print("ONNX Predictions")
print(ONNXOut)