# Python modules
import sys
import tensor_ipc_pb2
from google.protobuf.struct_pb2 import ListValue
import torch

# 3rd party modules
import posix_ipc

# Utils for this demo
import utils

def ListTensors(tensor_map):	
	for tensor in tensor_map.tensors:
		t0 = torch.as_tensor(tensor.rows[0])
		for index in range(1, len(tensor.rows)):
			t1 = torch.as_tensor(tensor.rows[index])
			t0 = torch.vstack((t0, t1))
		print(t0)

PY_MAJOR_VERSION = sys.version_info[0]

utils.say("Oooo 'ello, I'm Mrs. Conclusion!")

params = utils.read_params()

# Mrs. Premise has already created the message queue. I just need a handle
# to it.
mq = posix_ipc.MessageQueue(params["MESSAGE_QUEUE_NAME"])

tensor_map = tensor_ipc_pb2.TensorMap()

what_i_sent = ""

for i in range(0, params["ITERATIONS"]):
    utils.say("iteration %d" % i)
    s, _ = mq.receive()
    tensor_map.ParseFromString(s)
    ListTensors(tensor_map)
    #utils.say("Received %s" % s)


utils.say("")
utils.say("%d iterations complete" % (i + 1))
