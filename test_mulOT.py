#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from multiprocess_test_case import MultiProcessTestCase

import crypten.mpc.primitives.ot.baseOT as baseOT

from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
class TestObliviousTransfer(MultiProcessTestCase):
    def test_BaseOT(self):
        ot = baseOT.BaseOT((self.rank + 1) % self.world_size)
        self.length = 8
        if self.rank == 0:
            #Alice code her input x = 9 into choice in reverse order
            choices = [1, 0, 0, 1, 0, 0, 0, 0]
            self.assertEqual(len(choices), self.length)
            choices.append(0)
            msgcs = ot.receive(choices)
            SA = 0
            print("Alice receives messages: ", msgcs)
            for i in range(self.length):
                SA += int(msgcs[i])
            print("SA is ", SA)
            print("SB is ", msgcs[-1])
            S = (int(msgcs[-1]) + SA) % (2 ** self.length)
            print("S is ", S)
            self.assertEqual(S, 207)
        else:
            y = 23
            msg0s = []
            msg1s = []
            SB = 0
            for i in range(self.length):
                random_num = generate_kbit_random_tensor(size = (1,), bitlength = 32)
                msg0s.append(str(-random_num.item()))
                msg1s.append(str((y * (2 ** i) - random_num.item()) % (2 ** self.length)))
                SB += random_num.item()
            #Bob send the SB in the last entry
            msg0s.append(str(SB))
            msg1s.append(str(SB))
            ot.send(msg0s, msg1s)

if __name__ == "__main__":
    unittest.main()
