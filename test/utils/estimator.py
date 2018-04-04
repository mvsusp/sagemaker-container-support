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
import subprocess

from sagemaker.estimator import Framework
from sagemaker.fw_utils import create_image_uri

logger = logging.getLogger(__name__)


class TestEstimator(Framework):
    def __init__(self, name, framework_version, image_path, entry_point, py_version,
                 source_dir=None, hyperparameters=None, **kwargs):
        super(TestEstimator, self).__init__(entry_point, source_dir, hyperparameters, role='SageMakerRole',
                                            container_log_level=logging.DEBUG, **kwargs)
        self.image = None
        self.py_version = py_version
        self.framework_version = framework_version
        self.name = name
        self.image_path = image_path

    @classmethod
    def _from_training_job(cls, init_params, hyperparameters, image, sagemaker_session):
        pass

    def create_model(self, **kwargs):
        pass

    def train_image(self):
        tag = create_image_uri(self.sagemaker_session.boto_session.region_name, self.name,
                               self.train_instance_type, self.framework_version, self.py_version,
                               self.sagemaker_session.boto_session.client('sts').get_caller_identity()['Account'])

        self._build(tag)

        if self.train_instance_type.startswith('ml.'):
            self._push(tag)
        return tag

    def _build(self, tag):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.abspath(os.path.join(dir_path, '..', '..'))

        # TODO (MVS) - use shrlex to split
        subprocess.check_call('python setup.py bdist_wheel'.split(' '), cwd=root_path, stdout=self.stdout())

        device_type = 'gpu' if self.train_instance_type[:4] in ['ml.g', 'ml.p'] else 'cpu'

        dockerfile_name = 'Dockerfile.{}'.format(device_type)
        dockerfile_location = os.path.join(self.image_path, self.py_version, dockerfile_name)

        cmd = 'docker build -t {} -f {} .'.format(tag, dockerfile_location).split(' ')
        print(cmd)
        subprocess.check_call(cmd, cwd=root_path, stdout=self.stdout())
        print('created image {}'.format(tag))

    @staticmethod
    def stdout():
        return None if logger.isEnabledFor(logging.DEBUG) else subprocess.PIPE

    # TODO: mvs ecr repository creation when it does not exists
    def _push(self, tag):
        cmd = 'aws ecr get-login --no-include-email --region us-west-2'.split(' ')
        login = subprocess.check_output(cmd).strip()

        subprocess.check_call(login.split(' '), stdout=self.stdout())

        subprocess.check_call(cmd)
        cmd = 'docker push {}'.format(tag).split(' ')
        subprocess.check_call(cmd, stdout=self.stdout())
