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
import os
import subprocess
import sys
import time

import requests

import sagemaker_containers as smc


CURRENT_DIR = os.path.join(os.path.dirname(__file__))


def test_server():
    environ = os.environ.copy()
    environ[smc.environment.USE_NGINX_ENV] = 'false'

    application_path = os.path.join(CURRENT_DIR, 'simple_flask.py')
    process = subprocess.Popen(args=[sys.executable, application_path], env=environ)
    try:
        time.sleep(2)

        assert requests.get('http://127.0.0.1:8080/ping').status_code == 200
    finally:
        process.terminate()
        time.sleep(2)
