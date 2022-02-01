#! /usr/bin/python3

import tensor_ipc_pb2
from google.protobuf.struct_pb2 import ListValue
import sys
import torch

if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "file to write our tensors")
  sys.exit(-1)

dummy_input = torch.randn(5, 10, device="cpu")
print(dummy_input)
numlist = dummy_input.tolist()
# print(type(numlist))
# print(numlist)
lv = ListValue()
for i in range(len(numlist)):
	lv.add_list().extend(numlist[i])
print(len(lv))

tensor_map = tensor_ipc_pb2.TensorMap()
tensor1 = tensor_map.tensors.add()
tensor1.name = 'b-bias'
tensor1.rows.extend(lv)

assert tensor1.name == 'b-bias'
assert len(tensor1.rows)==5

# Write the new address book back to disk.
f = open(sys.argv[1], "wb")
f.write(tensor_map.SerializeToString())
# f.write(address_book.SerializeToString())
f.close()
