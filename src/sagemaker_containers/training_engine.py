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

import os
import traceback

import sagemaker_containers.training_environment as env
from sagemaker_containers.logger import create_logger

logger = create_logger()


SUCCESS_CODE = 0


class TrainingEngine(object):
    def __init__(self):
        self.train_fn = None
        self.environment = env.TrainingEnvironment()
        self.training_parameters = None

    def train(self):
        def decorator(train_fn):
            self.train_fn = train_fn
            return train_fn

        return decorator

    def run(self):
        # TODO: mvs env should be a property not a variable
        # training_environment = sagemaker_containers.training_environment.TrainingEnvironment()

        # TODO: mvs - print the training env in a prettier way
        logger.info("Started training with environment:")
        logger.info(str(self.environment))

        exit_code = SUCCESS_CODE
        try:
            self.environment.start_metrics_if_enabled()

            self.environment.download_user_module()
            user_module = self.environment.import_user_module()
            self.environment.load_training_parameters(user_module.train)
            self.train_fn(user_module, self.environment)

            self.environment.write_success_file()
        except Exception as e:
            trc = traceback.format_exc()
            message = 'Uncaught exception during training: {}\n{}\n'.format(e, trc)
            self.environment.write_failure_file(message)

            logger.error(message)
            exit_code = e.errno if hasattr(e, 'errno') else 1
            raise e
        finally:
            # Since threads in Python cannot be stopped, this is the only way to stop the application
            # https://stackoverflow.com/questions/9591350/what-is-difference-between-sys-exit0-and-os-exit0
            os._exit(exit_code)
