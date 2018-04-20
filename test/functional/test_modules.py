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

import logging
import os

import sagemaker
from sagemaker.fw_utils import tar_and_upload_dir

from sagemaker_containers import modules

logger = logging.getLogger(__name__)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)


def test_download_and_import(tmpdir):
    sagemaker_session = sagemaker.Session()
    b = sagemaker_session.default_bucket()

    file = tmpdir.join('myscript.py')
    file.write('print("hello")')

    setup = tmpdir.join('setup.py')
    setup.write("""
from distutils.core import setup

setup(name='myscript',
      version='1.0',
      py_modules=['myscript'],
      )
    """)

    uploaded_code = tar_and_upload_dir(session=sagemaker_session.boto_session,
                                       bucket=b,
                                       s3_key_prefix='test',
                                       script='myscript.py',
                                       directory=str(tmpdir))

    print(uploaded_code)
    module = modules.download_and_import(uploaded_code.s3_prefix, 'myscript')
