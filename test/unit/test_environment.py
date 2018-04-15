import json
import logging
from itertools import chain

from mock import Mock, patch

import pytest

from six import u
from six.moves import reload_module

import sagemaker_container_support.environment as environment

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

USER_HPS = dict(batch_size=32, learning_rate=.001)
SAGEMAKER_HPS = {'sagemaker_region': 'us-west-2', 'default_user_module_name': 'net',
                 'sagemaker_job_name': 'sagemaker-training-job', 'sagemaker_program': 'main.py',
                 'sagemaker_submit_directory': 'imagenet', 'sagemaker_enable_cloudwatch_metrics': True,
                 'sagemaker_container_log_level': logging.WARNING}

ALL_HPS = dict(chain(USER_HPS.items(), SAGEMAKER_HPS.items()))


@pytest.fixture(name='opt_ml_path')
def override_opt_ml_path(tmpdir):
    opt_ml = tmpdir.mkdir('opt').mkdir('ml')
    with patch.dict('os.environ', {'BASE_PATH': str(opt_ml)}):
        reload_module(environment)
        yield opt_ml
    reload_module(environment)


@pytest.fixture(name='input_path')
def override_input_path(opt_ml_path):
    return opt_ml_path.mkdir('input')


@pytest.fixture(name='input_config_path')
def override_input_config_path(input_path):
    return input_path.mkdir('config')


@pytest.fixture(name='input_data_path')
def override_input_data_path(input_path):
    return input_path.mkdir('data')


def test_read_json(tmpdir):
    file_path = write_json('hyperparameters.json', ALL_HPS, tmpdir)

    assert environment.read_json(file_path) == ALL_HPS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        environment.read_json('non-existent.json')


def test_read_hyperparameters(input_config_path):
    write_json('hyperparameters.json', ALL_HPS, input_config_path)

    assert environment.read_hyperparameters() == ALL_HPS


def test_read_key_serialized_hyperparameters(input_config_path):
    key_serialized_hps = {k: json.dumps(v) for k, v in ALL_HPS.items()}
    write_json('hyperparameters.json', key_serialized_hps, input_config_path)

    assert environment.read_hyperparameters() == ALL_HPS


def test_split_hyperparameters_only_provided_by_user():
    assert environment.split_hyperparameters(USER_HPS) == ({}, USER_HPS)


def test_split_hyperparameters_only_provided_by_sagemaker():
    assert environment.split_hyperparameters(SAGEMAKER_HPS) == (SAGEMAKER_HPS, {})


def test_split_hyperparameters():
    assert environment.split_hyperparameters(ALL_HPS) == (SAGEMAKER_HPS, USER_HPS)


def test_resource_config(input_config_path):
    write_json('resourceconfig.json', RESOURCE_CONFIG, input_config_path)

    assert environment.read_resource_config() == RESOURCE_CONFIG


def test_input_data_config(input_config_path):
    write_json('inputdataconfig.json', INPUT_DATA_CONFIG, input_config_path)

    assert environment.read_input_data_config() == INPUT_DATA_CONFIG


def test_channel_input_dirs(input_data_path):
    assert environment.channel_path('evaluation') == str(input_data_path.join('evaluation'))
    assert environment.channel_path('training') == str(input_data_path.join('training'))


@patch('subprocess.check_output', lambda s: u('GPU 0\nGPU 1'))
def test_gpu_count_in_gpu_instance():
    assert environment.gpu_count() == 2


def test_gpu_count_in_cpu_instance():
    assert environment.gpu_count() == 0


@patch('multiprocessing.cpu_count', lambda: 2)
def test_cpu_count():
    assert environment.cpu_count() == 2


@patch('sagemaker_container_support.environment.read_resource_config', lambda: RESOURCE_CONFIG)
@patch('sagemaker_container_support.environment.read_input_data_config', lambda: INPUT_DATA_CONFIG)
@patch('sagemaker_container_support.environment.read_hyperparameters', lambda: ALL_HPS)
@patch('sagemaker_container_support.environment.cpu_count', lambda: 8)
@patch('sagemaker_container_support.environment.gpu_count', lambda: 4)
def test_environment_create():
    env = environment.Environment.create(session=Mock())

    assert env.num_gpu == 4
    assert env.num_cpu == 8
    assert env.input_dir == '/opt/ml/input'
    assert env.input_config_dir == '/opt/ml/input/config'
    assert env.model_dir == '/opt/ml/model'
    assert env.output_dir == '/opt/ml/output'
    assert env.hyperparameters == USER_HPS
    assert env.resource_config == RESOURCE_CONFIG
    assert env.input_data_config == INPUT_DATA_CONFIG
    assert env.output_data_dir == '/opt/ml/output/data'
    assert env.hosts == RESOURCE_CONFIG['hosts']
    assert env.channel_input_dirs['train'] == '/opt/ml/input/data/train'
    assert env.channel_input_dirs['validation'] == '/opt/ml/input/data/validation'
    assert env.current_host == RESOURCE_CONFIG['current_host']
    assert env.module_name == 'main.py'
    assert env.module_dir == 'imagenet'
    assert env.enable_metrics
    assert env.log_level == logging.WARNING


def write_json(name, data, file_path):
    file_path = file_path.join(name)
    file_path.write(json.dumps(data))
    return str(file_path)
