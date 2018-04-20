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

from mock import call, patch, mock_open

import pip

import pytest
from six import PY2

from sagemaker_containers import modules


@patch('boto3.resource', autospec=True)
@pytest.mark.parametrize('url,bucket_name,key,dst', [
    ('S3://my-bucket/path/to/my-file', 'my-bucket', 'path/to/my-file', '/tmp/my-file'),
    ('s3://my-bucket/my-file', 'my-bucket', 'my-file', '/tmp/my-file')
])
def test_download(resource, url, bucket_name, key, dst):
    modules.download(url, dst)

    chain = call('s3').Bucket(bucket_name).download_file(key, dst)
    assert resource.mock_calls == chain.call_list()


def test_download_wrong_scheme():
    with pytest.raises(ValueError, message="Expecting 's3' scheme, got: c in c://my-bucket/my-file"):
        modules.download('c://my-bucket/my-file', '/tmp/file')


@patch('pip.main', autospec=True)
def test_install(main):
    main.return_value = pip.status_codes.SUCCESS
    modules.install('c://sagemaker-pytorch-container')

    main.assert_called_with(['install', 'c://sagemaker-pytorch-container'])


@patch('pip.main', lambda args: pip.status_codes.ERROR)
def test_install_fails():
    with pytest.raises(ValueError, message='Failed to install module git://aws/container-support with status code 1'):
        modules.install('git://aws/container-support')


builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('importlib.import_module')
@patch('sagemaker_containers.modules.install')
@patch('sagemaker_containers.modules.download')
@patch('tempfile.mkstemp')
@patch(builtins_open, mock_open())
@patch('tarfile.open')
def test_download_and_import_default_name(tar_open, mkstemp, download, install, import_module):
    mkstemp.side_effect = ['/tmp-1', '/tmp-2']

    module = modules.download_and_import('s3://bucket/my-module')

    open_call = open(download('s3://bucket/my-module', '/tmp-1'), 'rb')

    with tar_open(fileobj=open_call, mode='r:gz') as t:
        t.extractall.assert_called_with(path='/tmp-2')

    install.assert_called_with('/tmp-2')

    assert module == import_module(modules.DEFAULT_MODULE_NAME)


@patch('importlib.import_module')
@patch('sagemaker_containers.modules.install')
@patch('sagemaker_containers.modules.download')
@patch('tempfile.mkstemp')
@patch(builtins_open, mock_open())
@patch('tarfile.open')
def test_download_and_import(tar_open, mkstemp, download, install, import_module):
    mkstemp.side_effect = ['/tmp-1', '/tmp-2']

    module = modules.download_and_import('s3://bucket/my-module', 'another-module-name')

    open_call = open(download('s3://bucket/my-module', '/tmp-1'), 'rb')

    with tar_open(fileobj=open_call, mode='r:gz') as t:
        t.extractall.assert_called_with(path='/tmp-2')

    install.assert_called_with('/tmp-2')

    assert module == import_module('another-module-name')
