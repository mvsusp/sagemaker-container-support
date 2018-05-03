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
from mock import call, patch, PropertyMock

from sagemaker_containers.environment import ServingEnvironment
from sagemaker_containers.server import start

@patch.object(ServingEnvironment, 'model_server_workers', PropertyMock(return_value=2))
@patch.object(ServingEnvironment, 'model_server_timeout', PropertyMock(return_value=100))
@patch.object(ServingEnvironment, 'use_nginx', PropertyMock(return_value=False))
@patch('sagemaker_containers.environment.gpu_count', lambda: 0)
@patch('sys.exit', lambda x: 0)
@patch('subprocess.Popen')
def test_start_no_nginx(popen):
    calls = [call(
                 ['gunicorn',
                  '--timeout', '100',
                  '-k', 'gevent',
                  '-b', '0.0.0.0:8080',
                  '--worker-connections', '2000',
                  '-w', '2',
                  '--log-level', 'info',
                  'my_module'])]

    start('my_module')
    popen.assert_has_calls(calls)


@patch.object(ServingEnvironment, 'model_server_workers', PropertyMock(return_value=2))
@patch.object(ServingEnvironment, 'model_server_timeout', PropertyMock(return_value=100))
@patch.object(ServingEnvironment, 'use_nginx', PropertyMock(return_value=True))
@patch('pkg_resources.resource_filename', lambda x, y: '/tmp/nginx.conf')
@patch('sagemaker_containers.environment.gpu_count', lambda: 0)
@patch('sys.exit', lambda x: 0)
@patch('subprocess.Popen')
def test_start_with_nginx(popen):
    calls = [
        call(['nginx', '-c', '/tmp/nginx.conf']),
        call(['gunicorn',
              '--timeout', '100',
              '-k', 'gevent',
              '-b', 'unix:/tmp/gunicorn.sock',
              '--worker-connections', '2000',
              '-w', '2',
              '--log-level', 'info',
              'my_module'])
    ]
    start('my_module')
    popen.assert_has_calls(calls)
