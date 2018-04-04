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

import json
import logging

import pytest
from mock import patch, mock_open
from six import u, PY2

from sagemaker_containers.environment import ContainerEnvironment, parse_s3_url, download_s3_resource


@pytest.fixture(name='environment')
def fixture_environment():
    return ContainerEnvironment()


@patch('multiprocessing.cpu_count', lambda: 2)
def test_container_environment_default_settings():
    environment = ContainerEnvironment()

    assert environment.base_dir == '/opt/ml'
    assert environment.model_dir == '/opt/ml/model'
    assert environment.code_dir == '/opt/ml/code'
    assert environment.available_cpus == 2
    assert environment.available_gpus == 0
    assert not environment.user_script_name
    assert not environment.user_script_archive
    assert not environment.enable_cloudwatch_metrics
    assert environment.container_log_level == logging.INFO
    assert not environment.sagemaker_region


@patch('subprocess.check_output', lambda s: u('GPU 0\nGPU 1'))
def test_container_environment_multi_gpu():
    environment = ContainerEnvironment()

    assert environment.available_gpus == 2


def test_parse_s3_url():
    assert parse_s3_url('s3://bucket/key') == ('bucket', 'key')
    assert parse_s3_url('s3://bucket/prefix/key') == ('bucket', 'prefix/key')
    assert parse_s3_url('S3://bucket/prefix/key') == ('bucket', 'prefix/key')
    assert parse_s3_url('S3://bucket/prefix/key') == ('bucket', 'prefix/key')


def test_parse_s3_url_wrong_url():
    with pytest.raises(ValueError) as error:
        parse_s3_url('c://bucket/key')

    assert str(error.value) == "Expecting 's3' scheme, got: c in c://bucket/key"


# TODO (mvs) - test case when source does not exist
@patch('boto3.resource')
def test_download_s3_resource(resource):
    assert download_s3_resource('s3://bucket/key', 'c://target') == 'c://target'

    resource.assert_called_with('s3')
    bucket = resource().Bucket

    bucket.assert_called_with('bucket')
    bucket().download_file.assert_called_with('key', 'c://target')


# TODO (mvs) - test case handling exceptions
builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('sagemaker_containers.environment.download_s3_resource')
@patch(builtins_open, autospec=True)
@patch('tarfile.open', autospec=True)
@patch('tempfile.gettempdir', lambda: 'C://tmp')
def test_download_user_module(tar, open, download_s3, environment):
    scripts_path = 's3://sagemaker//scripts'
    environment.user_script_archive = scripts_path

    environment.download_user_module()

    tar_gz = 'C://tmp/script.tar.gz'
    download_s3.assert_called_with(scripts_path, tar_gz)
    open.assert_called_with(tar_gz, 'rb')
    tar.assert_called_with(mode='r:gz', fileobj=open().__enter__())
    tar().__enter__().extractall.assert_called_with(path='/opt/ml/code')


config = {'a': 1, 'b': {'c': 3}}


@patch(builtins_open, mock_open(read_data=json.dumps(config)))
def test_load_config(environment):
    assert environment.load_config('fake-path') == config


def raise_io_error(a, c):
    raise IOError()


@patch(builtins_open, raise_io_error)
def test_load_config_with_errors(environment):
    assert environment.load_config('fake-path') == {}
