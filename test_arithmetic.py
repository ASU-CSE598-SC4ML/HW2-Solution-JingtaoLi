#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import unittest
from multiprocess_test_case_10party import MultiProcessTestCase, get_random_test_tensor

import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
from crypten.common.rng import generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor
from crypten.common.util import count_wraps
from crypten.mpc.primitives import ArithmeticSharedTensor


class TestArithmetic(MultiProcessTestCase):
    """
    This class tests all functions of the ArithmeticSharedTensor.
    """

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communcator
        # print(self.rank)
        if self.rank == 0:
            crypten.init()

    def _check(self, encrypted_tensor, reference, msg, dst=None, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text(dst=dst)
        if dst is not None and dst != self.rank:
            self.assertIsNone(tensor)
            return

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def test_sum(self):
        """Tests sum reduction on encrypted tensor."""
        tensor_list = []
        encrypted_list = []
        for i in range(10):
            tensor_list.append(get_random_test_tensor(size = (1,), is_float=True))
            encrypted_list.append(ArithmeticSharedTensor(tensor_list[i]))
        if self.rank == 0:
            print("rank0 see tensor list as :" + str(tensor_list))
            print("rank0 see encrypted_list as :" + str(encrypted_list))

        if self.rank == 1:
            print("rank1 see tensor list as :" + str(tensor_list))
            print("rank1 see encrypted_list as :" + str(encrypted_list))


        reference = getattr(tensor_list[0], "add")(tensor_list[1])
        for i in range(8):
            reference = getattr(reference, "add")(tensor_list[i+2])

        encrypted_out = getattr(encrypted_list[0], "add")(encrypted_list[1])
        for i in range(8):
            encrypted_out = getattr(encrypted_out, "add")(encrypted_list[i+2])
            
        self._check(
                    encrypted_out,
                    reference,
                    "sum failed",
                )

# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
