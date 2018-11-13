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

import contextlib
import json
import os
import shutil
import tarfile
import tempfile

import boto3
from six.moves.urllib.parse import urlparse

from sagemaker_containers import _env, _params


def write_success_file():  # type: () -> None
    """Create a file 'success' when training is successful. This file doesn't need to have any content.
    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    """
    file_path = os.path.join(_env.output_dir, 'success')
    empty_content = ''
    write_file(file_path, empty_content)


def write_failure_file(failure_msg):  # type: (str) -> None
    """Create a file 'failure' if training fails after all algorithm output (for example, logging) completes,
    the failure description should be written to this file. In a DescribeTrainingJob response, Amazon SageMaker
    returns the first 1024 characters from this file as FailureReason.
    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    Args:
        failure_msg: The description of failure
    """
    file_path = os.path.join(_env.output_dir, 'failure')
    write_file(file_path, failure_msg)


@contextlib.contextmanager
def tmpdir(suffix='', prefix='tmp', dir=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the context exits.

    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix, otherwise there will be no
                        suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix; otherwise,
                        a default prefix is used.
        dir (str):  If dir is specified, the file will be created in that directory; otherwise, a default directory is
                        used.
    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    yield tmp
    shutil.rmtree(tmp)


def write_file(path, data, mode='w'):  # type: (str, str, str) -> None
    """Write data to a file.

    Args:
        path (str): path to the file.
        data (str): data to be written to the file.
        mode (str): mode which the file will be open.
    """
    with open(path, mode) as f:
        f.write(data)


def read_json(path):  # type: (str) -> dict
    """Read a JSON file.

    Args:
        path (str): Path to the file.

    Returns:
        (dict[object, object]): A dictionary representation of the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def s3_download(url, dst):  # type: (str, str) -> None
    """Download a file from S3.

    Args:
        url (str): the s3 url of the file.
        dst (str): the destination where the file will be saved.
    """
    url = urlparse(url)

    if url.scheme != 's3':
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    bucket, key = url.netloc, url.path.lstrip('/')

    region = os.environ.get('AWS_REGION', os.environ.get(_params.REGION_NAME_ENV))
    s3 = boto3.resource('s3', region_name=region)

    s3.Bucket(bucket).download_file(key, dst)


def download_and_extract(name, uri, path):  # type: (str, str, str) -> None
    """Download, prepare and install a compressed tar file from S3 or local directory as a module.

    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.

    This function downloads this compressed file, if provided, and transforms it as a module, and installs it.

    Args:
        uri (str): the location of the module.

    Returns:
        (module): the imported module
    """
    if uri.startswith('s3://'):
        with tmpdir() as tmp:

            dst = os.path.join(tmp, 'tar_file')
            s3_download(uri, dst)

            with tarfile.open(name=dst, mode='r:gz') as t:
                t.extractall(path=path)
    else:
        if os.path.isdir(uri):
            if os.path.exists(path):
                shutil.rmtree(path)

            shutil.copytree(uri, path)
        else:
            shutil.copy2(uri, os.path.join(path, name))
