# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import multiprocessing
import os

import sys

base_dir = 'opt/ml'


class Environment(object):
    """Provides access to common aspects of the container environment, including
    important system characteristics, filesystem locations, and configuration settings.
    """

    def __init__(self):
        self.base_dir = base_dir
        sys.path.insert(0, self.code_dir)

    @property
    def model_dir(self):
        return os.path.join(self.base_dir, 'model')

    @property
    def code_dir(self):
        """The directory where user-supplied code will be staged."""
        return os.path.join(self.base_dir, 'code')

    @property
    def available_cpus(self):
        """The number of cpus available in the current container."""
        return multiprocessing.cpu_count()

    @property
    def available_gpus(self):
        """The number of gpus available in the current container."""
        pass
