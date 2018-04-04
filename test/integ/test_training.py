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

import os
import tempfile

import numpy as np
import pytest

from utils.estimator import TestEstimator

TRAINING_DATA_FILENAME = 'training_data.npz'

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(dir_path, 'sagemaker-keras-containers/docker/2.1.3')


@pytest.fixture(name='training_channel', scope='session')
def fixture_training_channel(sagemaker_session):
    tmp_dir = tempfile.mkdtemp()

    features = np.random.random((1000, 20))
    labels = np.random.randint(10, size=(1000, 1))
    np.savez_compressed(os.path.join(tmp_dir, TRAINING_DATA_FILENAME), features=features, labels=labels)

    return sagemaker_session.upload_data(path=tmp_dir, key_prefix='integ-test-data/sagemaker-containers')


@pytest.mark.parametrize('py_version', ['py2', 'py3'])
@pytest.mark.parametrize('train_instance_type', ['local', 'ml.c4.xlarge', 'ml.p2.xlarge'])
def test_training(train_instance_type, py_version, training_channel):
    keras = TestEstimator(name='keras',
                          hyperparameters={'training_data_file': TRAINING_DATA_FILENAME},
                          framework_version='2.1.4',
                          image_path=image_path,
                          py_version=py_version,
                          train_instance_count=1,
                          train_instance_type=train_instance_type,
                          source_dir=os.path.join(dir_path, 'customer_scripts'),
                          entry_point='user_script_using_default_save.py')

    keras.fit({'training': training_channel})

    # TODO (mvs): assert container exit when functionality is available in local mode
    # TODO (mvs): check the logs when functionality is available in local mode
    if train_instance_type.startswith('local'):
        assert os.path.exists(os.path.join(keras.model_data, 'saved_model'))
    else:
        assert keras.model_data.endswith('output/model.tar.gz')
