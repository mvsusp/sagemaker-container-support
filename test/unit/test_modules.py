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

from mock import patch
import pytest

from sagemaker_containers import _errors, _modules


@contextlib.contextmanager
def patch_tmpdir():
    yield '/tmp'


@patch('importlib.import_module')
def test_exists(import_module):
    assert _modules.exists('my_module')

    import_module.side_effect = ImportError()

    assert not _modules.exists('my_module')


@patch('sagemaker_containers.training_env', lambda: {})
def test_run_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        _modules.run('wrong module')

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message


@patch('sagemaker_containers.entry_point.run')
def test_run(call):
    with pytest.warns(DeprecationWarning):
        _modules.run('pytest', ['--version'], {'PYTHONPATH': '1'}, False)
        call.assert_called_with('pytest.py', ['--version'], {'PYTHONPATH': '1'}, False)


@pytest.mark.parametrize('wait,cache', [[True, False], [True, False]])
@patch('sagemaker_containers.entry_point.run_from_uri')
def test_run_module_wait(run, wait, cache):
    with pytest.warns(DeprecationWarning):
        _modules.run_module(uri='s3://url', args=['42'], wait=wait, cache=cache)
        module_name = 'default_user_module_name.py'
        run.assert_called_with(args=['42'], env_vars=None, name=module_name, uri='s3://url', wait=wait)
