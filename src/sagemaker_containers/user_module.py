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
import importlib

import sys


class UserModule(object):
    def __init__(self, code_dir, framework_train_fn):
        self.code_dir = code_dir
        self.framework_train_fn = framework_train_fn
        self.user_script = 'user_script'

    def import_(self):
        sys.path.insert(0, self.code_dir)
        self.user_module = importlib.import_module(self.user_script)

    def train(self, training_environment):
        self.framework_train_fn(self.user_module, training_environment)