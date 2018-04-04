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

import importlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import tarfile

import boto3
from six.moves.urllib.parse import urlparse

logger = logging.getLogger(__name__)

BASE_DIRECTORY = "/opt/ml"
USER_SCRIPT_NAME_PARAM = "sagemaker_program"
USER_SCRIPT_ARCHIVE_PARAM = "sagemaker_submit_directory"
CLOUDWATCH_METRICS_PARAM = "sagemaker_enable_cloudwatch_metrics"
CONTAINER_LOG_LEVEL_PARAM = "sagemaker_container_log_level"
JOB_NAME_PARAM = "sagemaker_job_name"
CURRENT_HOST_ENV = "CURRENT_HOST"
JOB_NAME_ENV = "JOB_NAME"
USE_NGINX_ENV = "SAGEMAKER_USE_NGINX"
SAGEMAKER_REGION_PARAM_NAME = 'sagemaker_region'
FRAMEWORK_MODULE_NAME = "CONTAINER_MODULE_NAME"


class ContainerEnvironment(object):
    """Provides access to common aspects of the container environment, including
    important system characteristics, filesystem locations, and configuration settings.
    """

    def __init__(self, base_dir=BASE_DIRECTORY):
        self._session = boto3.Session()

        self.base_dir = base_dir
        "The current root directory for SageMaker interactions (``/opt/ml`` when running in SageMaker)."

        self.model_dir = os.path.join(base_dir, "model")
        "The directory to write model artifacts to so they can be handed off to SageMaker."

        self.code_dir = os.path.join(base_dir, "code")
        "The directory where user-supplied code will be staged."

        self.available_cpus = self._get_available_cpus()
        "The number of cpus available in the current container."

        self.available_gpus = self._get_available_gpus()
        "The number of gpus available in the current container."

        # subclasses will override
        self.user_script_name = None
        "The filename of the python script that contains user-supplied training/hosting code."

        # subclasses will override
        self.user_script_archive = None
        "The S3 location of the python code archive that contains user-supplied training/hosting code"

        self.enable_cloudwatch_metrics = False
        "Report system metrics to CloudWatch? (default = False)"

        # subclasses will override
        self.container_log_level = logging.INFO
        "The logging level for the root logger."

        # subclasses will override
        self.sagemaker_region = None
        "The current AWS region."

    def download_user_module(self):
        """Download user-supplied python archive from S3.
        """
        tmp = os.path.join(tempfile.gettempdir(), "script.tar.gz")
        download_s3_resource(self.user_script_archive, tmp)
        with open(tmp, 'rb') as f:
            with tarfile.open(mode='r:gz', fileobj=f) as t:
                t.extractall(path=self.code_dir)

    def import_user_module(self):
        """Import user-supplied python module.
        """
        sys.path.insert(0, self.code_dir)

        script = self.user_script_name
        if script.endswith(".py"):
            script = script[:-3]

        user_module = importlib.import_module(script)
        return user_module

    def start_metrics_if_enabled(self):
        if self.enable_cloudwatch_metrics:
            logger.info("starting metrics service")
            subprocess.Popen(['telegraf', '--config', '/usr/local/etc/telegraf.conf'])

    @staticmethod
    def _get_available_cpus():
        return multiprocessing.cpu_count()

    @staticmethod
    def _get_available_gpus():
        gpus = 0
        try:
            output = str(subprocess.check_output(["nvidia-smi", "--list-gpus"]))
            gpus = sum([1 for x in output.split('\n') if x.startswith('GPU ')])
        except Exception as e:
            logger.warning("No GPUs detected (normal if no gpus installed): %s" % str(e))

        return gpus

    @staticmethod
    def load_config(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)

        except IOError:
            # TODO (mvs): training environment should work outside the container.
            # We need to determine default beha viour
            return {}


def parse_s3_url(url):
    """ Returns an (s3 bucket, key name/prefix) tuple from a url with an s3 scheme
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: {} in {}".format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip('/')


def download_s3_resource(source, target):
    """ Downloads the s3 object source and stores in a new file with path target.
    """
    logger.info("Downloading {} to {}".format(source, target))
    s3 = boto3.resource('s3')

    script_bucket_name, script_key_name = parse_s3_url(source)
    script_bucket = s3.Bucket(script_bucket_name)
    script_bucket.download_file(script_key_name, target)

    return target
