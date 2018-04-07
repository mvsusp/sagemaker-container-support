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

import logging
import multiprocessing
import os
import shlex
import subprocess
import sys

logger = logging.getLogger(__name__)

# TODO (mvs) -  create a shortcut decorator to allow get/set outside the class, e.g. env.model_dir = 'x'
base_dir = '/opt/ml'


class Environment(object):
    """Provides access to common aspects of the container environment, including
    important system characteristics, filesystem locations, environment variables and configuration settings.
    """

    def __init__(self, base_directory=None):
        """Construct an `Environment` instance.

        Args:
            base_directory (string): the base directory where the environment will read/write files.
                The default base directory in SageMaker is '/opt/ml'
        """
        self.base_dir = base_directory if base_directory else base_dir
        self._model_dir = os.path.join(self.base_dir, 'model')
        self._code_dir = os.path.join(self.base_dir, 'code')
        self._available_cpus = multiprocessing.cpu_count()
        self._available_gpus = get_available_gpus()

        # Adds the code directory to PYTHONPATH
        sys.path.insert(0, self.code_dir)

    @property
    def model_dir(self):
        """Your algorithm should write all final model artifacts to this directory.
        Amazon SageMaker copies this data as a single object in compressed tar format
        to the S3 location that you specified in the CreateTrainingJob request.

        Amazon SageMaker aggregates the result in a tar file and uploads to s3.

        Returns:
            (string): the directory where models should be saved, e.g., /opt/ml/model/
        """
        return self._model_dir

    @property
    def code_dir(self):
        """When a SageMaker training jobs starts, the user provided training script
        (or python package) will be saved in this folder.

        Returns:
            (string): the directory containing the user provided training script (or python package)
        """
        return self._code_dir

    @property
    def available_cpus(self):
        """The number of cpus available in the current container.

        Returns:
            (int): number of cpus available in the current container.
        """
        return self._available_cpus

    @property
    def available_gpus(self):
        """The number of gpus available in the current container.

        Returns:
            (int): number of gpus available in the current container.
        """
        return self._available_gpus


def get_available_gpus():
    """The number of gpus available in the current container.

    Returns:
        (int): number of gpus available in the current container.
    """
    try:
        cmd = shlex.split('nvidia-smi --list-gpus')
        output = str(subprocess.check_output(cmd))
        return sum([1 for x in output.split('\n') if x.startswith('GPU ')])
    except OSError:
        logger.warning("No GPUs detected (normal if no gpus installed)")
        return 0
