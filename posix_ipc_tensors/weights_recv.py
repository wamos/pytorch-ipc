import socket
import os

import tensor_ipc_pb2
from google.protobuf.struct_pb2 import ListValue
import torch

if os.path.exists("/tmp/tensor_unix_sockets"):
    os.remove("/tmp/tensor_unix_sockets")

print("Opening socket...")
server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
server.bind("/tmp/tensor_unix_sockets")
max_iters=10
current_iter=0
expect_bytes = int(571) # 571 to be exact
print(type(expect_bytes))

print("Listening...")
# for SOCK_STREAM, you'll need these two
#server.listen(1)
#connection, client_address = server.accept()

while current_iter < max_iters:
	buf = bytes()
	while len(buf) < expect_bytes: 	
		weights = server.recv(expect_bytes)
		buf = buf + weights
		print("len(buf):", len(buf))

	print("after recv while loop len(buf)", len(buf))
	print("current-iter:", current_iter)	
	current_iter+=1

print("-" * 20)
print("total_len", total_len)
print("Shutting down...")
server.close()
os.remove("/tmp/tensor_unix_sockets")
print("Done")