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
import os
import shlex
import subprocess

import sagemaker_containers.training_environment as env

_PYTHONPATH = 'PYTHONPATH'


class Training(object):
    """Training class to start training in the container.

    When training starts, the `Training` object ensures that the user script (or Python package)
    is imported and executed in a different process, reporting success in the end of the training.

    `Training` is responsible for error handling and metrics reporting and reports failures in case
    of exceptions.
    """

    def __init__(self, training_process_fn):
        """Constructs a `Training` instance.

        When a training starts, e.g. `Training(training_process_fn).run()`, the `Training` instance
        will execute the following steps:

            1 - install required Python dependencies
            2 - start required customs metrics
            3 - load training environment
            4 - download the user script (or Python package) containing the functions required by the framework
            5 - start training process by invoking `training_process_fn`
            6 - report success/failure

        Args:
            training_process_fn: a training process function that will be executed by the `Training`. Signature:

                * Args:
                    * `training_environment`(TrainingEnvironment): an instance of the `TrainingEnvironment`.
        """
        self.environment = env.TrainingEnvironment()
        self.training_process_fn = training_process_fn

    # TODO(mvsusp) - add update_training_environment_fn to allow frameworks to update the training env.
    @classmethod
    def from_train_fn(cls, train_fn):
        """Creates a Training instance from a `train_fn`.

        Args:
            train_fn: training function implemented by a framework. Follows the signature:

                * Args:
                    * `user_module`(module): the user provided training script imported as a module. Contains
                        the training functions required by the framework, e.g.: train(...) and save_model(...).
                    * `training_environment`(TrainingEnvironment): an instance of the `TrainingEnvironment`.

        Returns:
            Training
        """

        def import_user_module_and_invoke_train_inside_process(training_environment):
            """This training process fn imports the user module and invokes train_fn with the user
            module and the training environment as parameters inside a process

            Args:
                training_environment: an instance of the `TrainingEnvironment`.
            """
            user_module = importlib.import_module(training_environment.user_script_name)

            # TODO (mvsusp): execute this function inside a process with subprocess
            train_fn(user_module, training_environment)

        return cls(training_process_fn=import_user_module_and_invoke_train_inside_process)

    @classmethod
    def from_module(cls, module_name):
        """Creates a Training instance from module already registered in the PYTHONPATH.

        Args:
            module_name (string): the name of the module to be imported.

        Returns:
            Training
        """

        def invoke_module_inside_process(training_environment):
            """This training process fn executes the user module in a different process

            Args:
                training_environment: an instance of the `TrainingEnvironment`.
            """
            env_vars = os.environ.copy()
            env_vars[_PYTHONPATH] = '%s:%s' % (training_environment.code_dir, env_vars.get(_PYTHONPATH, ''))

            cmd = shlex.split('python -m %s' % module_name)

            # TODO (mvsusp): pass in the env vars and PYTHONPATH
            subprocess.check_call(cmd, env=env_vars)

        return cls(training_process_fn=invoke_module_inside_process)

    def run(self):
        """Starts the training process.

        Execute the following steps:

        1 - install required python dependencies
        2 - start required customs metrics
        3 - load training environment
        4 - start training process by invoking `training_process_fn`
        5 - report success/failure
        """

        self.training_process_fn(self.environment)

        try:
            # TODO(mvsusp) - install required python dependencies
            # TODO(mvsusp) - start required customs metrics

            self.training_process_fn(self.environment)
            self.environment.write_success_file()
        except subprocess.CalledProcessError as e:
            # TODO(mvsusp) - add traceback
            self.environment.write_failure_file('Uncaught exception during training: %s' % e)
            raise e
