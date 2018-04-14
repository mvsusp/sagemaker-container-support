# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import logging
import multiprocessing
import os
import shlex
import subprocess

import boto3

from six import PY2, reraise

if PY2:
    JSONDecodeError = None
else:
    from json.decoder import JSONDecodeError

logging.basicConfig()
logger = logging.getLogger(__name__)

BASE_PATH_ENV = 'BASE_PATH'
CURRENT_HOST_ENV = 'CURRENT_HOST'
JOB_NAME_ENV = 'JOB_NAME'
USE_NGINX_ENV = 'SAGEMAKER_USE_NGINX'

BASE_PATH = os.environ.get(BASE_PATH_ENV, '/opt/ml')

MODEL_PATH = os.path.join(BASE_PATH, 'model')
INPUT_PATH = os.path.join(BASE_PATH, 'input')
INPUT_DATA_PATH = os.path.join(INPUT_PATH, 'data')
INPUT_CONFIG_PATH = os.path.join(INPUT_PATH, 'config')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, 'data')

HYPERPARAMETERS_FILE = 'hyperparameters.json'
RESOURCE_CONFIG_FILE = 'resourceconfig.json'
INPUT_DATA_CONFIG_FILE = 'inputdataconfig.json'

PROGRAM_PARAM = 'sagemaker_program'
SUBMIT_DIR_PARAM = 'sagemaker_submit_directory'
ENABLE_METRICS_PARAM = 'sagemaker_enable_cloudwatch_metrics'
LOG_LEVEL_PARAM = 'sagemaker_container_log_level'
JOB_NAME_PARAM = 'sagemaker_job_name'
DEFAULT_MODULE_NAME_PARAM = 'default_user_module_name'
REGION_PARAM_NAME = 'sagemaker_region'

SAGEMAKER_HPS = [PROGRAM_PARAM, SUBMIT_DIR_PARAM, ENABLE_METRICS_PARAM, REGION_PARAM_NAME,
                 LOG_LEVEL_PARAM, JOB_NAME_PARAM, DEFAULT_MODULE_NAME_PARAM]


def read_json(path):
    """Read a JSON file.

    Args:
        path (string): path to the file

    Returns:
        A Python dictionary representation of the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def read_hyperparameters():
    """Read the hyperparameters from /opt/ml/input/config/hyperparameters.json.


    For more information about hyperparameters.json:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-hyperparameters

    Returns:
         hyperparameters (dict[string, object]): a map containing the hyperparameters.

    """
    hyperparameters = read_json(os.path.join(INPUT_CONFIG_PATH, HYPERPARAMETERS_FILE))

    try:
        return {k: json.loads(v) for k, v in hyperparameters.items()}
    except (JSONDecodeError, TypeError):
        return hyperparameters
    except ValueError as e:
        if str(e) == 'No JSON object could be decoded':
            logger.warning("Failed to parse hyperparameters' values to Json. Returning the hyperparameters instead:")
            logging.warning(hyperparameters)
            return hyperparameters
        reraise(e)


def split_hyperparameters(hyperparameters, keys=SAGEMAKER_HPS):
    """Split a dictionary in two by the provided keys. The default key SAGEMAKER_HPS splits user provided
    hyperparameters from SageMaker Python SDK provided hyperparameters.

    Args:
        hyperparameters: A Python dictionary
        keys: Lists of keys which will be the split criteria

    Returns:
        criteria (dict[string, object]), not_criteria (dict[string, object]): the result of the split criteria.
    """
    not_keys = set(hyperparameters.keys()) - set(keys)

    return {k: hyperparameters[k] for k in keys if k in hyperparameters}, {k: hyperparameters[k] for k in not_keys}


def read_resource_config():
    """Read the resource configuration from /opt/ml/input/config/resourceconfig.json.

    For more information about resourceconfig.json:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

    Returns:
        resource_config: A dict<string, string> with the contents from /opt/ml/input/config/resourceconfig.json.
                        It has the following keys:
                            - current_host: The name of the current container on the container network.
                                For example, 'algo-1'.
                            -  hosts: The list of names of all containers on the container network,
                                sorted lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']`
                                for a three-node cluster.
    """
    return read_json(os.path.join(INPUT_CONFIG_PATH, RESOURCE_CONFIG_FILE))


def read_input_data_config():
    """Read the input data configuration from /opt/ml/input/config/inputdataconfig.json.

        For more information about inpudataconfig.json:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

    Returns:
            input_data_config: A dict<string, string> with the contents from /opt/ml/input/config/inputdataconfig.json.

                                For example, suppose that you specify three data channels (train, evaluation, and
                                validation) in your request. This dictionary will contain:

                                {'train': {
                                    'ContentType':  'trainingContentType',
                                    'TrainingInputMode': 'File',
                                    'S3DistributionType': 'FullyReplicated',
                                    'RecordWrapperType': 'None'
                                },
                                'evaluation' : {
                                    'ContentType': 'evalContentType',
                                    'TrainingInputMode': 'File',
                                    'S3DistributionType': 'FullyReplicated',
                                    'RecordWrapperType': 'None'
                                },
                                'validation': {
                                    'TrainingInputMode': 'File',
                                    'S3DistributionType': 'FullyReplicated',
                                    'RecordWrapperType': 'None'
                                }}

                                You can find more information about /opt/ml/input/config/inputdataconfig.json here:
                                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig
    """
    return read_json(os.path.join(INPUT_CONFIG_PATH, INPUT_DATA_CONFIG_FILE))


def channel_input_dirs(channel):
    """ Returns the directory containing the channel data file(s) which is:
    - <self.base_dir>/input/data/<channel>

    For more information about channels: https://docs.aws.amazon.com/sagemaker/latest/dg/API_Channel.html

    Returns:
        (str) The input data directory for the specified channel.
    """
    return os.path.join(INPUT_DATA_PATH, channel)


def num_gpu():
    """The number of gpus available in the current container.

    Returns:
        (int): number of gpus available in the current container.
    """
    try:
        cmd = shlex.split('nvidia-smi --list-gpus')
        output = str(subprocess.check_output(cmd))
        return sum([1 for x in output.split('\n') if x.startswith('GPU ')])
    except OSError:
        logger.warning('No GPUs detected (normal if no gpus installed)')
        return 0


def num_cpu():
    """The number of cpus available in the current container.

    Returns:
        (int): number of cpus available in the current container.
    """
    return multiprocessing.cpu_count()


class Environment(object):
    """Provides access to aspects of the training environment relevant to training jobs, including
    hyperparameters, system characteristics, filesystem locations, environment variables and configuration settings.

    Example on how a script can use training environment:
        ```
        >>>import sagemaker_container_support as cs
        >>>env = cs.Environment.create()

        get the path of the channel 'training' from the inputdataconfig.json file
        >>>training_dir = env.channel_input_dirs['training']

        get a the hyperparameter 'training_data_file' from hyperparameters.json file
        >>>file_name = env.hyperparameters['training_data_file']

        # get the folder where the model should be saved
        >>>model_dir = env.model_dir

        >>>data = np.load(os.path.join(training_dir, training_data_file))

        >>>x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

        >>>model = ResNet50(weights='imagenet')

        unfreeze the model to allow fine tuning
        ...

        >>>model.fit(x_train, y_train)

        save the model in the end of training
        >>>model.save(os.path.join(model_dir, 'saved_model'))
        ```
    """

    def __init__(self, input_dir, input_config_dir, model_dir, output_dir, hyperparameters, resource_config,
                 input_data_config, output_data_dir, hosts, channel_input_dirs, current_host, num_gpu, num_cpu,
                 module_name,
                 module_dir, enable_metrics, log_level):
        """

        Args:
            input_dir: The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
                        and configuration files before and during training. The input data directory has the
                        following subdirectories: config (`input_config_dir`) and data (`input_data_dir`)

            input_config_dir: The directory where standard SageMaker configuration files are located,
                        e.g. /opt/ml/input/config/.

                        SageMaker training creates the following files in this folder when training starts:
                            - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob
                                    request available in this file.
                            - `inputdataconfig.json`: You specify data channel information in the InputDataConfig
                                    parameter in a CreateTrainingJob request. Amazon SageMaker makes this information
                                    available in this file.
                            - `resourceconfig.json`: name of the current host and all host containers in the training

                        More information about these files can be find here:
                            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

            model_dir:  the directory where models should be saved, e.g., /opt/ml/model/

            output_dir: The directory where training success/failure indications will be written, e.g. /opt/ml/output.
                        To save non-model artifacts check `output_data_dir`.

            hyperparameters: An instance of `HyperParameters` containing the training job hyperparameters.

            resource_config: A dict<string, string> with the contents from /opt/ml/input/config/resourceconfig.json.
                            It has the following keys:
                                - current_host: The name of the current container on the container network.
                                    For example, 'algo-1'.
                                -  hosts: The list of names of all containers on the container network,
                                    sorted lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']`
                                    for a three-node cluster.

            input_data_config: A dict<string, string> with the contents from /opt/ml/input/config/inputdataconfig.json.

                                For example, suppose that you specify three data channels (train, evaluation, and
                                validation) in your request. This dictionary will contain:

                                {'train': {
                                    'ContentType':  'trainingContentType',
                                    'TrainingInputMode': 'File',
                                    'S3DistributionType': 'FullyReplicated',
                                    'RecordWrapperType': 'None'
                                },
                                'evaluation' : {
                                    'ContentType': 'evalContentType',
                                    'TrainingInputMode': 'File',
                                    'S3DistributionType': 'FullyReplicated',
                                    'RecordWrapperType': 'None'
                                },
                                'validation': {
                                    'TrainingInputMode': 'File',
                                    'S3DistributionType': 'FullyReplicated',
                                    'RecordWrapperType': 'None'
                                }}

                                You can find more information about /opt/ml/input/config/inputdataconfig.json here:
                                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig

            output_data_dir: The dir to write non-model training artifacts (e.g. evaluation results) which will be
                        retained by SageMaker, e.g. /opt/ml/output/data.

                        As your algorithm runs in a container, it generates output including the status of the
                        training job and model and output artifacts. Your algorithm should write this information
                        to the this directory.

            hosts: The list of names of all containers on the container network, sorted lexicographically.
                    For example, `['algo-1', 'algo-2', 'algo-3']` for a three-node cluster.

            channel_input_dirs:   A dict[string, string] containing the data channels and the directories where the
                            training data was saved.

                            When you run training, you can partition your training data into different logical
                            'channels'. Depending on your problem, some common channel ideas are: 'train', 'test',
                             'evaluation' or 'images','labels'.

                            The format of channel_input_dir is as follows:

                                - `channel`[key] - the name of the channel defined in the input_data_config.
                                - `training data path`[value] - the path to the directory where the training data is
                                saved.

            current_host: The name of the current container on the container network. For example, 'algo-1'.

            num_gpu: The number of gpus available in the current container.

            num_cpu: The number of cpus available in the current container.

        Returns:
            A `TrainerEnvironment` object.
        """
        self.input_dir = input_dir
        self.input_config_dir = input_config_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.hyperparameters = hyperparameters
        self.resource_config = resource_config
        self.input_data_config = input_data_config
        self.output_data_dir = output_data_dir
        self.hosts = hosts
        self.channel_input_dirs = channel_input_dirs
        self.current_host = current_host
        self.num_gpu = num_gpu
        self.num_cpu = num_cpu
        self.module_name = module_name
        self.module_dir = module_dir
        self.enable_metrics = enable_metrics
        self.log_level = log_level

    @classmethod
    def create(cls, session=None):
        """
        Returns: an instance of `Environment`
        """
        session = session if session else boto3.Session()

        resource_config = read_resource_config()
        current_host = resource_config['current_host']
        hosts = resource_config['hosts']

        input_data_config = read_input_data_config()

        sagemaker_hyperparameters, hyperparameters = split_hyperparameters(read_hyperparameters())

        sagemaker_region = sagemaker_hyperparameters.get(REGION_PARAM_NAME, session.region_name)

        os.environ[JOB_NAME_ENV] = sagemaker_hyperparameters[JOB_NAME_PARAM]
        os.environ[CURRENT_HOST_ENV] = current_host
        os.environ[REGION_PARAM_NAME.upper()] = sagemaker_region

        return cls(input_dir=INPUT_PATH,
                   input_config_dir=INPUT_CONFIG_PATH,
                   model_dir=MODEL_PATH,
                   output_dir=OUTPUT_PATH,
                   output_data_dir=OUTPUT_DATA_PATH,
                   current_host=current_host,
                   hosts=hosts,
                   channel_input_dirs={channel: channel_input_dirs(channel) for channel in input_data_config},
                   num_gpu=num_gpu(),
                   num_cpu=num_cpu(),
                   hyperparameters=hyperparameters,
                   resource_config=resource_config,
                   input_data_config=read_input_data_config(),
                   module_name=sagemaker_hyperparameters.get(PROGRAM_PARAM),
                   module_dir=sagemaker_hyperparameters.get(SUBMIT_DIR_PARAM),
                   enable_metrics=sagemaker_hyperparameters.get(ENABLE_METRICS_PARAM, False),
                   log_level=sagemaker_hyperparameters.get(LOG_LEVEL_PARAM, logging.INFO)
                   )
