#! /usr/bin/python3

import tensor_ipc_pb2
import sys
from google.protobuf.struct_pb2 import ListValue
import torch

def ListTensors(tensor_map):	
	for tensor in tensor_map.tensors:
		for row in tensor.rows:
			t1 =torch.as_tensor(row)
			print(t1)

if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "file to read our tensors")
  sys.exit(-1)

tensor_map = tensor_ipc_pb2.TensorMap()

# Read the existing file
f = open(sys.argv[1], "rb")
tensor_map.ParseFromString(f.read())
f.close()

ListTensors(tensor_map)
