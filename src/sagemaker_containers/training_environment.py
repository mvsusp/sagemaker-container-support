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

from sagemaker_containers.environment import Environment


class TrainingEnvironment(Environment):
    """Provides access to aspects of the container environment relevant to training jobs.
    """

    @property
    def input_dir(self):
        """The base directory for training data and configuration files."""
        return os.path.join(self.base_dir, 'input')

    @property
    def input_config_dir(self):
        """The directory where standard SageMaker configuration files are located."""
        return os.path.join(self.base_dir, 'input/config')

    @property
    def output_dir(self):
        """The directory where training success/failure indications will be written."""
        return os.path.join(self.base_dir, 'output')

    @property
    def resource_config(self):
        """The dict of resource configuration settings."""
        pass

    @property
    def hyperparameters(self):
        """The dict of hyperparameters that were passed to the CreateTrainingJob API."""
        pass

    @property
    def current_host(self):
        """The hostname of the current container."""
        pass

    @property
    def hosts(self):
        """The list of hostnames available to the current training job."""
        pass

    @property
    def output_data_dir(self):
        """The dir to write non-model training artifacts (e.g. evaluation results) which will be retained by
        SageMaker. """
        pass

    @property
    def channels(self):
        """The dict of training input data channel name to directory with the input files for that channel."""
        pass

    @property
    def channel_dirs(self):
        """"""
        pass

    def load_training_parameters(self, fn):
        pass

    def write_success_file(self):
        pass

    def write_failure_file(self, message):
        pass
