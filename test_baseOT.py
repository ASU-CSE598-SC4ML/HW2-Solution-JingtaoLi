#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from multiprocess_test_case import MultiProcessTestCase

import crypten.mpc.primitives.ot.baseOT as baseOT


class TestObliviousTransfer(MultiProcessTestCase):
    def test_BaseOT(self):
        ot = baseOT.BaseOT((self.rank + 1) % self.world_size)
        if self.rank == 0:
            choices = [1]
            msgcs = ot.receive(choices)
            print(msgcs)
            self.assertEqual(msgcs, ["xyz"])
        else:
            # play the role of receiver first with choice bit [1, 0]
            msg0s = ["abc"]
            msg1s = ["xyz"]
            ot.send(msg0s, msg1s)

if __name__ == "__main__":
    unittest.main()
