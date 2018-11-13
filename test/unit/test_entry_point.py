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

import contextlib
import os
import sys


from mock import MagicMock, patch
import pytest
from six import PY2

from sagemaker_containers import _env, _errors, entry_point

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('sagemaker_containers.entry_point._check_error', autospec=True)
def test_install(check_error):
    path = 'c://sagemaker-pytorch-container'
    entry_point.install(path)

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_error.assert_called_with(cmd, _errors.InstallModuleError, cwd=path)

    with patch('os.path.exists', return_value=True):
        entry_point.install(path)

        check_error.assert_called_with(cmd + ['-r', 'requirements.txt'], _errors.InstallModuleError, cwd=path)


@patch('sagemaker_containers.entry_point._check_error', autospec=True)
def test_install_fails(check_error):
    check_error.side_effect = _errors.ClientError()
    with pytest.raises(_errors.ClientError):
        entry_point.install('git://aws/container-support')


@patch('sys.executable', None)
def test_install_no_python_executable():
    with pytest.raises(RuntimeError) as e:
        entry_point.install('git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@contextlib.contextmanager
def patch_tmpdir():
    yield '/tmp'


@patch('sagemaker_containers.training_env', lambda: {})
@patch('subprocess.Popen', lambda *args, **kwargs: MagicMock(wait=MagicMock(return_value=32)))
def test_call_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        entry_point.call('wrong module')

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message


def test_python_executable_exception():
    with patch('sys.executable', None):
        with pytest.raises(RuntimeError):
            entry_point._python_executable()


@patch('sagemaker_containers.training_env', lambda: {})
def test_call():
    entry_point.call('pytest', ['--version'])


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers.entry_point.call')
def test_run_wait(call, download_and_extract):
    entry_point.run('train.py', uri='s3://url', args=['42'])

    download_and_extract.assert_called_with('train.py', 's3://url', _env.code_dir)
    call.assert_called_with('train.py', ['42'], {}, True)


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers.entry_point.call')
@patch('os.chmod')
def test_run_no_wait(chmod, call, download_and_extract):
    entry_point.run('launcher.sh', uri='s3://url', args=['42'], wait=False)

    download_and_extract.assert_called_with('launcher.sh', 's3://url', _env.code_dir)
    call.assert_called_with('launcher.sh', ['42'], {}, False)
    chmod.assert_called_with(os.path.join(_env.code_dir, 'launcher.sh'), 755)
