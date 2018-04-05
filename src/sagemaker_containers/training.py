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

import importlib
import subprocess

import sys

import sagemaker_containers.training_environment as env


class Training(object):
    def __init__(self, train_fn=None, module_name=None):
        self.environment = env.TrainingEnvironment()
        self.train_fn = train_fn
        self.module_name = module_name

    @classmethod
    def from_train_fn(cls, train_fn):
        return cls(train_fn=train_fn)

    @classmethod
    def from_module(cls, module_name):
        return cls(module_name=module_name)

    def start(self):

        try:
            if self.train_fn:

                user_module = self.import_user_module(self.environment.code_dir)

                self.train_fn(user_module, self.environment)
            else:
                subprocess.Popen(['python', '-m', self.module_name])

            self.environment.write_success_file()
        except Exception as e:
            self.environment.write_failure_file('Uncaught exception during training: %s' % e)
            raise e

    def import_user_module(self, code_dir, user_script='user_script'):
        return importlib.import_module(user_script)
