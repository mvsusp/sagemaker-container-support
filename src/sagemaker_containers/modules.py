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
import os
import shlex
import shutil
import tarfile
import tempfile

import boto3
import pip

from six.moves.urllib.parse import urlparse

DEFAULT_MODULE_NAME = 'default_user_module_name'


def download(url, dst):
    url = urlparse(url)

    if url.scheme != 's3':
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, dst))

    bucket, key = url.netloc, url.path.lstrip('/')

    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(key, dst)


def install(path):
    print(pip.main(shlex.split('freeze')))

    result = pip.main(shlex.split('install %s -U' % path))
    if result != pip.status_codes.SUCCESS:
        raise ValueError('Failed to install module %s with status code %s' % (path, result))


def download_and_import(url, name=DEFAULT_MODULE_NAME):
    with tempfile.NamedTemporaryFile() as tmp:
        download(url, tmp.name)

        with open(tmp.name, 'rb') as f:
            with tarfile.open(mode='r:gz', fileobj=f) as t:
                tmpdir = tempfile.mkdtemp()
                try:
                    t.extractall(path=tmpdir)

                    install(tmpdir)

                    return importlib.import_module(name)
                finally:
                    shutil.rmtree(tmpdir)