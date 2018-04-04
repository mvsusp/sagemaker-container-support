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
import inspect
import json
import logging
import os

import six
import yaml

from sagemaker_containers.environment import ContainerEnvironment
from sagemaker_containers.environment import BASE_DIRECTORY, USER_SCRIPT_NAME_PARAM, USER_SCRIPT_ARCHIVE_PARAM, \
    CLOUDWATCH_METRICS_PARAM, CONTAINER_LOG_LEVEL_PARAM, JOB_NAME_PARAM, JOB_NAME_ENV, CURRENT_HOST_ENV, \
    SAGEMAKER_REGION_PARAM_NAME
from sagemaker_containers.logger import create_logger, set_logger_level

HYPERPARAMETERS_FILE = "hyperparameters.json"
RESOURCE_CONFIG_FILE = "resourceconfig.json"
INPUT_DATA_CONFIG_FILE = "inputdataconfig.json"
S3_URI_PARAM = 'sagemaker_s3_uri'

logger = create_logger()


# TODO: mvs - review docstrings
class TrainingEnvironment(ContainerEnvironment):
    """Provides access to aspects of the container environment relevant to training jobs.
    """

    def __init__(self, base_dir=BASE_DIRECTORY):
        super(TrainingEnvironment, self).__init__(base_dir)
        self.input_dir = os.path.join(self.base_dir, "input")
        "The base directory for training data and configuration files."

        self.input_config_dir = os.path.join(self.input_dir, "config")
        "The directory where standard SageMaker configuration files are located."

        self.output_dir = os.path.join(self.base_dir, "output")
        "The directory where training success/failure indications will be written."

        self.resource_config = self.load_config(os.path.join(self.input_config_dir, RESOURCE_CONFIG_FILE))
        "The dict of resource configuration settings."

        self.hyperparameters = self._load_hyperparameters(os.path.join(self.input_config_dir, HYPERPARAMETERS_FILE))
        "The dict of hyperparameters that were passed to the CreateTrainingJob API."

        # TODO: change default.
        self.network_interface_name = self.resource_config.get('network_interface_name', 'ethwe')
        "The name of the network interface to use for distributed training."

        self.current_host = self.resource_config.get('current_host', '')
        "The hostname of the current container."

        self.hosts = self.resource_config.get('hosts', [])
        "The list of hostnames available to the current training job."

        self.output_data_dir = os.path.join(self.output_dir, "data", self.current_host if len(self.hosts) > 1 else '')
        "The dir to write non-model training artifacts (e.g. evaluation results) which will be retained by SageMaker. "

        # TODO validate docstring
        self.channels = self.load_config(os.path.join(self.input_config_dir, INPUT_DATA_CONFIG_FILE))
        "The dict of training input data channel name to directory with the input files for that channel."

        # TODO validate docstring
        self.channel_dirs = {channel: self._get_channel_dir(channel) for channel in self.channels}

        self.user_script_name = self.hyperparameters.get(USER_SCRIPT_NAME_PARAM, '')
        self.user_script_archive = self.hyperparameters.get(USER_SCRIPT_ARCHIVE_PARAM, '')

        self.enable_cloudwatch_metrics = self.hyperparameters.get(CLOUDWATCH_METRICS_PARAM, False)
        self.container_log_level = self.hyperparameters.get(CONTAINER_LOG_LEVEL_PARAM, logging.INFO)

        set_logger_level(self.container_log_level)

        os.environ[JOB_NAME_ENV] = self.hyperparameters.get(JOB_NAME_PARAM, '')
        os.environ[CURRENT_HOST_ENV] = self.current_host

        self.sagemaker_region = self.hyperparameters.get(SAGEMAKER_REGION_PARAM_NAME, self._session.region_name)
        os.environ[SAGEMAKER_REGION_PARAM_NAME.upper()] = self.sagemaker_region

        self.distributed = len(self.hosts) > 1

        self.kwargs_for_training = {
            'hyperparameters': dict(self.hyperparameters),
            'input_data_config': dict(self.channels),
            'channel_input_dirs': dict(self.channel_dirs),
            'output_data_dir': self.output_data_dir,
            'model_dir': self.model_dir,
            'num_gpus': self.available_gpus,
            'num_cpus': self.available_cpus,
            'hosts': list(self.hosts),
            'current_host': self.current_host
        }
        """ Returns a dictionary of key-word arguments for input to the user supplied module train function. """
        self.training_parameters = None

    def __str__(self):
        # TODO (mvs) - print only public attrs and unit test this function
        def only_public_attrs(attr_name):
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                return False
            return True

        public_attrs = filter(only_public_attrs, dir(self))

        return yaml.safe_dump({attr_name: getattr(self, attr_name) for attr_name in public_attrs})

    def load_training_parameters(self, fn):
        self.training_parameters = self.matching_parameters(fn)

    def matching_parameters(self, fn):
        args, keywords = get_fn_args_and_keywords(fn)
        # avoid forcing our callers to specify **kwargs in their function
        # signature. If they have **kwargs we still pass all the args, but otherwise
        # we will just pass what they ask for.
        if keywords is None:
            kwargs_to_pass = {}
            for arg in args:
                if arg != "self" and arg in self.kwargs_for_training:
                    kwargs_to_pass[arg] = self.kwargs_for_training[arg]
        else:
            kwargs_to_pass = self.kwargs_for_training
        return kwargs_to_pass

    def _load_hyperparameters(self, path):
        serialized = self.load_config(path)
        return self._deserialize_hyperparameters(serialized)

    # TODO expecting serialized hyperparams might break containers that aren't launched by python sdk
    @staticmethod
    def _deserialize_hyperparameters(hp):
        return {k: json.loads(v) for (k, v) in hp.items()}

    def write_success_file(self):
        self.write_output_file('success')

    def write_failure_file(self, message):
        self.write_output_file('failure', message)

    def write_output_file(self, file_name, message=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        message = message if message else 'Training finished with {}'.format(file_name)

        with open(os.path.join(self.output_dir, file_name), 'a') as fd:
            fd.write(message)

    def _get_channel_dir(self, channel):
        """ Returns the directory containing the channel data file(s).

        This is either:

        - <self.base_dir>/input/data/<channel> OR
        - <self.base_dir>/input/data/<channel>/<channel_s3_suffix>

        Where channel_s3_suffix is the hyperparameter value with key <S3_URI_PARAM>_<channel>.

        The first option is returned if <self.base_dir>/input/data/<channel>/<channel_s3_suffix>
        does not exist in the file-system or <S3_URI_PARAM>_<channel> does not exist in
        self.hyperparmeters. Otherwise, the second option is returned.

        TODO: Refactor once EASE downloads directly into /opt/ml/input/data/<channel>
        TODO: Adapt for Pipe Mode

        Returns:
            (str) The input data directory for the specified channel.
        """
        channel_s3_uri_param = "{}_{}".format(S3_URI_PARAM, channel)
        if channel_s3_uri_param in self.hyperparameters:
            channel_s3_suffix = self.hyperparameters.get(channel_s3_uri_param)
            channel_dir = os.path.join(self.input_dir, 'data', channel, channel_s3_suffix)
            if os.path.exists(channel_dir):
                return channel_dir
        return os.path.join(self.input_dir, 'data', channel)


def get_fn_args_and_keywords(func):
    if six.PY2:
        arg_spec = inspect.getargspec(func)
        return arg_spec.args, arg_spec.keywords

    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    keyworks = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keyworks = keyworks[0] if keyworks else None
    return args, keyworks
