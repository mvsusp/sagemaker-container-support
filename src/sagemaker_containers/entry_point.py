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

import enum
import os
import shlex
import subprocess
import sys


import six
from six import string_types

from sagemaker_containers import _env, _errors, _files, _logging

logger = _logging.get_logger()


class CommandType(enum.Enum):
    PYTHON_PACKAGE = 1
    PYTHON_PROGRAM = 2
    COMMAND = 3


def command_type(name):  # type: (str) -> CommandType
    if 'setup.py' in os.listdir(_env.code_dir):
        return CommandType.PYTHON_PACKAGE
    elif name.endswith('.py'):
        return CommandType.PYTHON_PROGRAM
    else:
        return CommandType.COMMAND


def has_requirements(path):  # type: (str) -> None
    return os.path.exists(os.path.join(path, 'requirements.txt'))


def install_requirements(path):  # type: (str) -> None
    if has_requirements(path):
        cmd = '%s -m pip install -r requirements.txt' % _python_executable()

        logger.info('Installing requirements.txt with the following command:\n%s', cmd)

        _check_error(cmd, _errors.InstallModuleError, cwd=path)


def run(name, uri, args, env_vars=None, wait=True):
    # type: (str, str, list, dict, bool) -> subprocess.Popen
    """Download, prepare and executes a compressed tar file from S3 or provided directory as a module.

    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, transforms it as a module, and executes it.
    Args:
        uri (str): the location of the module.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        name (str): name of the script or module.
        wait (bool): If True run_module will wait for the user module to exit and check the exit code,
                     otherwise it will launch the user module with subprocess and return the process object.
    """
    env_vars = (env_vars or {}).copy()
    code_dir = _env.code_dir

    download_and_install(name, uri, code_dir)

    _env.write_env_vars(env_vars)

    return call(name, args, env_vars, wait)


def download_and_install(name, uri, dst):
    if not os.listdir(dst):
        _files.download_and_extract(name, uri, dst)
    sys.path.insert(0, dst)
    entry_point_type = command_type(name)
    if entry_point_type is CommandType.PYTHON_PACKAGE:
        install(dst)
    else:
        install_requirements(dst)
    if entry_point_type is CommandType.COMMAND:
        os.chmod(os.path.join(dst, name), 755)


def install(path):  # type: (str) -> None
    """Install a Python module in the executing Python environment.

    Args:
        path (str):  Real path location of the Python module.
    """
    cmd = '%s -m pip install -U . ' % _python_executable()

    if has_requirements(path):
        cmd += '-r requirements.txt'

    logger.info('Installing module with the following command:\n%s', cmd)

    _check_error(shlex.split(cmd), _errors.InstallModuleError, cwd=path)


def call(name, args=None, env_vars=None, wait=True):  # type: (str, list, dict, bool) -> subprocess.Popen
    """Run Python module as a script.

    Search sys.path for the named module and execute its contents as the __main__ module.

    Since the argument is a module name, you must not give a file extension (.py). The module name should be a valid
    absolute Python module name, but the implementation may not always enforce this (e.g. it may allow you to use a name
    that includes a hyphen).

    Package names (including namespace packages) are also permitted. When a package name is supplied instead of a
    normal module, the interpreter will execute <pkg>.__main__ as the main module. This behaviour is deliberately
    similar to the handling of directories and zip files that are passed to the interpreter as the script argument.

    Note This option cannot be used with built-in modules and extension modules written in C, since they do not have
    Python module files. However, it can still be used for precompiled modules, even if the original source file is
    not available. If this option is given, the first element of sys.argv will be the full path to the module file (
    while the module file is being located, the first element will be set to "-m"). As with the -c option,
    the current directory will be added to the start of sys.path.

    You can find more information at https://docs.python.org/3/using/cmdline.html#cmdoption-m

    Example:

        >>>import sagemaker_containers
        >>>from sagemaker_containers.beta.framework import mapping, modules

        >>>env = sagemaker_containers.training_env()
        {'channel-input-dirs': {'training': '/opt/ml/input/training'}, 'model_dir': '/opt/ml/model', ...}


        >>>hyperparameters = env.hyperparameters
        {'batch-size': 128, 'model_dir': '/opt/ml/model'}

        >>>args = mapping.to_cmd_args(hyperparameters)
        ['--batch-size', '128', '--model_dir', '/opt/ml/model']

        >>>env_vars = mapping.to_env_vars()
        ['SAGEMAKER_CHANNELS':'training', 'SAGEMAKER_CHANNEL_TRAINING':'/opt/ml/input/training',
        'MODEL_DIR':'/opt/ml/model', ...}

        >>>modules.run('user_script', args, env_vars)
        SAGEMAKER_CHANNELS=training SAGEMAKER_CHANNEL_TRAINING=/opt/ml/input/training \
        SAGEMAKER_MODEL_DIR=/opt/ml/model python -m user_script --batch-size 128 --model_dir /opt/ml/model

    Args:
        name (str): module name in the same format required by python -m <module-name> cli command.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        wait (bool): If True run_module will wait for the user module to exit and check the exit code,
                     otherwise it will launch the user module with subprocess and return the process object.
    """
    args = args or []
    env_vars = env_vars or {}

    cmd_by_entry_point_type = {
        CommandType.PYTHON_PACKAGE: [_python_executable(), '-m', name.replace('.py', '')],
        CommandType.PYTHON_PROGRAM: [_python_executable(), name],
        CommandType.COMMAND:        [name],
    }

    cmd = cmd_by_entry_point_type[command_type(name)] + args

    _logging.log_script_invocation(cmd, env_vars)

    previous = os.getcwd()
    try:
        os.chdir(_env.code_dir)

        if wait:
            return _check_error(cmd, _errors.ExecuteUserScriptError)
        else:
            return _popen(cmd)
    finally:
        os.chdir(previous)


def _check_error(cmd, error_class, **kwargs):
    cmd = shlex.split(cmd) if isinstance(cmd, string_types) else cmd

    try:
        process = _popen(cmd, **kwargs)
        return_code = process.wait()

        if return_code:
            raise error_class(return_code=return_code, cmd=' '.join(cmd))
        return process
    except Exception as e:
        six.reraise(error_class, error_class(e), sys.exc_info()[2])


def _popen(cmd, **kwargs):
    return subprocess.Popen(cmd, env=os.environ, **kwargs)


def _python_executable():
    """Returns the real path for the Python executable, if it exists. Returns RuntimeError otherwise.

    Returns:
        (str): the real path of the current Python executable
    """
    if not sys.executable:
        raise RuntimeError('Failed to retrieve the real path for the Python executable binary')
    return sys.executable
