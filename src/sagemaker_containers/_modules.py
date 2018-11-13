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

import importlib
import subprocess  # noqa: F401 imported but unused
import sys
import warnings

import six

from sagemaker_containers import _env, _errors, _logging, entry_point

logger = _logging.get_logger()

DEFAULT_MODULE_NAME = 'default_user_module_name'


def exists(name):  # type: (str) -> bool
    """Return True if the module exists. Return False otherwise.

    Args:
        name (str): module name.

    Returns:
        (bool): boolean indicating if the module exists or not.
    """
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    else:
        return True


def run(module_name, args=None, env_vars=None, wait=True):  # type: (str, list, dict, bool) -> Popen
    """Run Python module as a script.

    Search sys.path for the named module and execute its contents as the __main__ module.

    Since the argument is a module name, you must not give a file extension (.py). The module name should be a valid
    absolute Python module name, but the implementation may not always enforce this (e.g. it may allow you to use a name
    that includes a hyphen).

    Package names (including namespace packages) are also permitted. When a package name is supplied instead of a
    normal module, the interpreter will execute <pkg>.__main__ as the main module. This behaviour is deliberately
    similar to the handling of directories and zipfiles that are passed to the interpreter as the script argument.

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
        module_name (str): module name in the same format required by python -m <module-name> cli command.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
    """
    msg = 'run is now deprecated and will be removed in the future. Use entry_point.run instead'
    warnings.warn(msg, DeprecationWarning)
    return entry_point.run(_user_program_name(module_name), args, env_vars, wait)


def import_module(uri, name=DEFAULT_MODULE_NAME, cache=None):  # type: (str, str, bool) -> module
    """Download, prepare and install a compressed tar file from S3 or provided directory as a module.
    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, if provided, and transforms it as a module, and installs it.
    Args:
        name (str): name of the script or module.
        uri (str): the location of the module.
        cache (bool): default True. It will not download and install the module again if it is already installed.
    Returns:
        (module): the imported module
    """
    _warning_cache_deprecation(cache)
    entry_point.download_and_install(name=_user_program_name(name), uri=uri, dst=_env.code_dir)

    try:
        module = importlib.import_module(name)
        six.moves.reload_module(module)

        return module
    except Exception as e:
        six.reraise(_errors.ImportModuleError, _errors.ImportModuleError(e), sys.exc_info()[2])


def run_module(uri, args, env_vars=None, name=DEFAULT_MODULE_NAME, cache=True, wait=True):
    # type: (str, list, dict, str, bool, bool) -> subprocess.Popen
    """Download, prepare and executes a compressed tar file from S3 or provided directory as a module.

    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, transforms it as a module, and executes it.
    Args:
        uri (str): the location of the module.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        name (str): name of the script or module.
        cache (bool): If True it will avoid downloading the module again, if already installed.
        wait (bool): If True run_module will wait for the user module to exit and check the exit code,
                     otherwise it will launch the user module with subprocess and return the process object.
    """
    _warning_cache_deprecation(cache)
    msg = 'run_module is now deprecated and will be removed in the future. Use entry_point.run_from_uri instead'
    warnings.warn(msg, DeprecationWarning)
    return entry_point.run_from_uri(uri=uri, args=args, env_vars=env_vars, name=_user_program_name(name), wait=wait)


def _user_program_name(name):
    return "%s.py" % name


def _warning_cache_deprecation(cache):
    if cache is not None:
        msg = 'the cache parameter is unnecessary anymore. Cache is always set to True'
        warnings.warn(msg, DeprecationWarning)
