# Python modules
import time
import sys
import hashlib

import tensor_ipc_pb2
from google.protobuf.struct_pb2 import ListValue
import torch

# 3rd party modules
import posix_ipc

# Utils for this demo
import utils

PY_MAJOR_VERSION = sys.version_info[0]

utils.say("Oooo 'ello, I'm Mrs. Premise!")

params = utils.read_params()

# Create the message queue.
mq = posix_ipc.MessageQueue(params["MESSAGE_QUEUE_NAME"], posix_ipc.O_CREX)

dummy_input = torch.randn(5, 10, device="cpu")
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
assert len(tensor1.rows)==5
s=tensor_map.SerializeToString()

# The first message is a random string (the current time).
# s = time.asctime()
# utils.say("Sending %s" % s)
mq.send(s)
what_i_sent = s

for i in range(0, params["ITERATIONS"]):
    utils.say("iteration %d" % i)
    print(dummy_input)
    mq.send(s)

utils.say("")
utils.say("%d iterations complete" % (i + 1))

utils.say("Destroying the message queue.")
mq.close()
# I could call simply mq.unlink() here but in order to demonstrate
# unlinking at the module level I'll do it that way.
posix_ipc.unlink_message_queue(params["MESSAGE_QUEUE_NAME"])
