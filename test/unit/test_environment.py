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
import itertools
import json
import logging
import os

from mock import Mock, mock_open, patch
import pytest
import six

from sagemaker_containers import env
import test

builtins_open = '__builtin__.open' if six.PY2 else 'builtins.open'

RESOURCE_CONFIG = dict(current_host='algo-1', hosts=['algo-1', 'algo-2', 'algo-3'])

INPUT_DATA_CONFIG = {
    'train': {
        'ContentType': 'trainingContentType',
        'TrainingInputMode': 'File',
        'S3DistributionType': 'FullyReplicated',
        'RecordWrapperType': 'None'
    },
    'validation': {
        'TrainingInputMode': 'File',
        'S3DistributionType': 'FullyReplicated',
        'RecordWrapperType': 'None'
    }}

USER_HYPERPARAMETERS = dict(batch_size=32, learning_rate=.001)
SAGEMAKER_HYPERPARAMETERS = {'sagemaker_region': 'us-west-2', 'default_user_module_name': 'net',
                             'sagemaker_job_name': 'sagemaker-training-job', 'sagemaker_program': 'main.py',
                             'sagemaker_submit_directory': 'imagenet', 'sagemaker_enable_cloudwatch_metrics': True,
                             'sagemaker_container_log_level': logging.WARNING}

ALL_HYPERPARAMETERS = dict(itertools.chain(USER_HYPERPARAMETERS.items(), SAGEMAKER_HYPERPARAMETERS.items()))


def test_read_json():
    test.write_json(ALL_HYPERPARAMETERS, env.HYPERPARAMETERS_PATH)

    assert env.read_json(env.HYPERPARAMETERS_PATH) == ALL_HYPERPARAMETERS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        env.read_json('non-existent.json')


def test_read_hyperparameters():
    test.write_json(ALL_HYPERPARAMETERS, env.HYPERPARAMETERS_PATH)

    assert env.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_read_key_serialized_hyperparameters():
    key_serialized_hps = {k: json.dumps(v) for k, v in ALL_HYPERPARAMETERS.items()}
    test.write_json(key_serialized_hps, env.HYPERPARAMETERS_PATH)

    assert env.read_hyperparameters() == ALL_HYPERPARAMETERS


@patch('sagemaker_containers.env.read_json', lambda x: {'a': 1})
@patch('json.loads')
def test_read_exception(loads):
    loads.side_effect = ValueError('Unable to read.')

    assert env.read_hyperparameters() == {'a': 1}


def test_resource_config():
    test.write_json(RESOURCE_CONFIG, env.RESOURCE_CONFIG_PATH)

    assert env.read_resource_config() == RESOURCE_CONFIG


def test_input_data_config():
    test.write_json(INPUT_DATA_CONFIG, env.INPUT_DATA_CONFIG_FILE_PATH)

    assert env.read_input_data_config() == INPUT_DATA_CONFIG


def test_channel_input_dirs():
    input_data_path = env.INPUT_DATA_PATH
    assert env.channel_path('evaluation') == os.path.join(input_data_path, 'evaluation')
    assert env.channel_path('training') == os.path.join(input_data_path, 'training')


@patch('subprocess.check_output', lambda s: b'GPU 0\nGPU 1')
def test_gpu_count_in_gpu_instance():
    assert env.gpu_count() == 2


@patch('multiprocessing.cpu_count', lambda: OSError())
def test_gpu_count_in_cpu_instance():
    assert env.gpu_count() == 0


@patch('multiprocessing.cpu_count', lambda: 2)
def test_cpu_count():
    assert env.cpu_count() == 2


@pytest.fixture(name='training_env')
def create_training_env():
    with patch('sagemaker_containers.env.read_resource_config', lambda: RESOURCE_CONFIG), \
         patch('sagemaker_containers.env.read_input_data_config', lambda: INPUT_DATA_CONFIG), \
         patch('sagemaker_containers.env.read_hyperparameters', lambda: ALL_HYPERPARAMETERS), \
         patch('sagemaker_containers.env.cpu_count', lambda: 8), \
         patch('sagemaker_containers.env.gpu_count', lambda: 4):
        session_mock = Mock()
        session_mock.region_name = 'us-west-2'
        return env.TrainingEnv()


@pytest.fixture(name='serving_env')
def create_serving_env():
    with patch('sagemaker_containers.env.cpu_count', lambda: 8), \
         patch('sagemaker_containers.env.gpu_count', lambda: 4):
        os.environ[env.USE_NGINX_ENV] = 'false'
        os.environ[env.MODEL_SERVER_TIMEOUT_ENV] = '20'
        os.environ[env.CURRENT_HOST_ENV] = 'algo-1'
        os.environ[env.USER_PROGRAM_ENV] = 'main.py'
        os.environ[env.SUBMIT_DIR_ENV] = 'my_dir'
        os.environ[env.ENABLE_METRICS_ENV] = 'true'
        os.environ[env.REGION_NAME_ENV] = 'us-west-2'
        return env.ServingEnv()


def test_train_env(training_env):
    assert training_env.num_gpus == 4
    assert training_env.num_cpus == 8
    assert training_env.input_dir.endswith('/opt/ml/input')
    assert training_env.input_config_dir.endswith('/opt/ml/input/config')
    assert training_env.model_dir.endswith('/opt/ml/model')
    assert training_env.output_dir.endswith('/opt/ml/output')
    assert training_env.hyperparameters == USER_HYPERPARAMETERS
    assert training_env.resource_config == RESOURCE_CONFIG
    assert training_env.input_data_config == INPUT_DATA_CONFIG
    assert training_env.output_data_dir.endswith('/opt/ml/output/data/algo-1')
    assert training_env.hosts == RESOURCE_CONFIG['hosts']
    assert training_env.channel_input_dirs['train'].endswith('/opt/ml/input/data/train')
    assert training_env.channel_input_dirs['validation'].endswith('/opt/ml/input/data/validation')
    assert training_env.current_host == RESOURCE_CONFIG['current_host']
    assert training_env.module_name == 'main'
    assert training_env.module_dir == 'imagenet'
    assert training_env.enable_metrics
    assert training_env.log_level == logging.WARNING
    assert training_env.network_interface_name == 'ethwe'


def test_serving_env(serving_env):
    assert serving_env.num_gpus == 4
    assert serving_env.num_cpus == 8
    assert serving_env.use_nginx is False
    assert serving_env.model_server_timeout == 20
    assert serving_env.model_server_workers == 8
    assert serving_env.module_name == 'main'
    assert serving_env.enable_metrics
    assert serving_env.framework_module is None


def test_train_env_properties(training_env):
    assert training_env.properties() == ['channel_input_dirs', 'current_host', 'enable_metrics', 'framework_module',
                                         'hosts', 'hyperparameters', 'input_config_dir', 'input_data_config',
                                         'input_dir', 'log_level', 'model_dir', 'module_dir', 'module_name',
                                         'network_interface_name', 'num_cpus', 'num_gpus', 'output_data_dir',
                                         'output_dir', 'resource_config']


def test_serving_env_properties(serving_env):
    print(serving_env.properties())
    assert serving_env.properties() == ['current_host', 'enable_metrics', 'framework_module', 'log_level', 'model_dir',
                                        'model_server_timeout', 'model_server_workers', 'module_dir', 'module_name',
                                        'num_cpus', 'num_gpus', 'use_nginx']


def test_request_properties(serving_env):
    print(serving_env.properties())
    assert serving_env.properties() == ['current_host', 'enable_metrics', 'framework_module', 'log_level', 'model_dir',
                                        'model_server_timeout', 'model_server_workers', 'module_dir', 'module_name',
                                        'num_cpus', 'num_gpus', 'use_nginx']


@patch('sagemaker_containers.env.cpu_count', lambda: 8)
@patch('sagemaker_containers.env.gpu_count', lambda: 4)
def test_env_dictionary():
    session_mock = Mock()
    session_mock.region_name = 'us-west-2'
    os.environ[env.USER_PROGRAM_ENV] = 'my_app.py'
    _env = env.Env()

    assert len(_env) == len(_env.properties())

    assert _env['num_gpus'] == 4
    assert _env['num_cpus'] == 8
    assert _env['module_name'] == 'my_app'
    assert _env['enable_metrics']
    assert _env['log_level'] == logging.INFO


@pytest.mark.parametrize('sagemaker_program', ['program.py', 'program'])
def test_env_module_name(sagemaker_program):
    session_mock = Mock()
    session_mock.region_name = 'us-west-2'
    os.environ[env.USER_PROGRAM_ENV] = sagemaker_program
    assert env.Env().module_name == 'program'


@patch('tempfile.mkdtemp')
@patch('shutil.rmtree')
def test_tmpdir(rmtree, mkdtemp):
    with env.tmpdir():
        mkdtemp.assert_called()
    rmtree.assert_called()


@patch('tempfile.mkdtemp')
@patch('shutil.rmtree')
def test_tmpdir_with_args(rmtree, mkdtemp):
    with env.tmpdir('suffix', 'prefix', '/tmp'):
        mkdtemp.assert_called_with(dir='/tmp', prefix='prefix', suffix='suffix')
    rmtree.assert_called()


@patch(builtins_open, mock_open())
def test_write_file():
    env.write_file('/tmp/my-file', '42')

    open.assert_called_with('/tmp/my-file', 'w')

    open().write.assert_called_with('42')

    env.write_file('/tmp/my-file', '42', 'x')

    open.assert_called_with('/tmp/my-file', 'x')

    open().write.assert_called_with('42')
