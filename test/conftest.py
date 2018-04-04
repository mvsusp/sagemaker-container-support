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
import logging
import os
import platform

import boto3
import pytest
import shutil
import tempfile

from sagemaker import Session

logger = logging.getLogger(__name__)

logging.getLogger('sagemaker').setLevel(logging.DEBUG)
# logging.getLogger('sagemaker.image').setLevel(logging.DEBUG)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


@pytest.fixture
def tmp_dir():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    tmp = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield tmp

    shutil.rmtree(tmp, True)


@pytest.fixture(scope='session')
def sagemaker_session():
    boto_session = boto3.Session(region_name=DEFAULT_REGION)

    return Session(boto_session=boto_session)
