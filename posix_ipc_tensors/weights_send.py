# -*- coding: utf-8 -*-
import socket
import os

import tensor_ipc_pb2
from google.protobuf.struct_pb2 import ListValue
import torch

print("Connecting...")
if os.path.exists("/tmp/tensor_unix_sockets"):
	client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
	client.connect("/tmp/tensor_unix_sockets")
	max_iters=10
	current_iter=0

	while current_iter < max_iters:
		print("current-iter:", current_iter)
		dummy_input = torch.randn(100, 100, device="cpu")
		print(dummy_input)
		numlist = dummy_input.tolist()
		lv = ListValue()
		for i in range(len(numlist)):
			lv.add_list().extend(numlist[i])

		tensor_map = tensor_ipc_pb2.TensorMap()
		tensor1 = tensor_map.tensors.add()
		tensor1.name = 'b-bias'
		tensor1.rows.extend(lv)

		assert tensor1.name == 'b-bias'
		assert len(tensor1.rows)==100
		s = tensor_map.SerializeToString()
		print(len(s))
		vs = memoryview(s)

		# print("sending", len(s))
		total_bytes = 0
		sent_iter = 0
		packet_size = 8192
		while total_bytes < len(s):
			start_index = 0 + packet_size*sent_iter
			end_index = packet_size*(sent_iter+1) if packet_size*(sent_iter+1) < len(s) else len(s)
			print("start", start_index, "end", end_index)
			sent_bytes = client.send(vs[start_index:end_index])
			total_bytes += sent_bytes
			print("sent", sent_bytes, "on sent_iter", sent_iter, "total bytes", total_bytes)
			sent_iter += 1
						
		# start_index = 0 + 100*sent_iter
		# sent_bytes = client.send(vs[start_index:])
		# total_bytes += sent_bytes
		# print("sent", sent_bytes, "on sent_iter", sent_iter, "total bytes", total_bytes)
		#print("sent", sent_bytes)
		#total_bytes += sent_bytes
		current_iter+=1

	print("Shutting down.")
	client.close()

else:
    print("Couldn't Connect!")
    print("Done")