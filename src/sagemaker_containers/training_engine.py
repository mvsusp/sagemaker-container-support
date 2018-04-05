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

import sagemaker_containers.training_environment as env
from sagemaker_containers.user_module import UserModule


class TrainingEngine(object):
    def __init__(self, framework_train=None):
        self.framework_train = framework_train
        self.environment = env.TrainingEnvironment()

    def framework_train_fn(self):
        def decorator(train_fn):
            self.framework_train = train_fn
            return train_fn

        return decorator

    def run(self):

        try:
            user_module = UserModule(self.environment.code_dir, self.framework_train)

            user_module.import_()

            user_module.train(self.environment)

            self.environment.write_success_file()
        except Exception as e:
            self.environment.write_failure_file('Uncaught exception during training: %s' % e)
            raise e
