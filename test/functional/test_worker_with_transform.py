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

import json

from mock import patch, PropertyMock
import pytest
from six.moves import range

import sagemaker_containers as smc


class DummyTransformer(object):
    def __init__(self):
        self.calls = dict(initialize=0, transform=0)

    def initialize(self):
        self.calls['initialize'] += 1

    def transform(self):
        self.calls['transform'] += 1
        return smc.worker.TransformSpec(json.dumps(self.calls), smc.content_types.APPLICATION_JSON)


@patch('sagemaker_containers.environment.ServingEnvironment.module_name', PropertyMock(return_value='user_program'))
@pytest.mark.parametrize('module_name,expected_name', [('my_module', 'my_module'), (None, 'user_program')])
def test_worker(module_name, expected_name):
    transformer = DummyTransformer()

    with smc.worker.run(transformer, module_name=module_name).test_client() as worker:
        assert worker.application.import_name == expected_name

        assert worker.get('/ping').status_code == smc.status_codes.OK

        for _ in range(9):
            response = worker.post('/invocations')
            assert response.status_code == smc.status_codes.OK

        response = worker.post('/invocations')
        assert json.loads(response.get_data().decode('utf-8')) == dict(initialize=1, transform=10)
        assert response.mimetype == smc.content_types.APPLICATION_JSON


def test_worker_with_custom_ping():
    transformer = DummyTransformer()

    def custom_ping():
        return 'ping', smc.status_codes.ACCEPTED

    with smc.worker.run(transformer, custom_ping, 'custom_ping').test_client() as worker:
        response = worker.get('/ping')
        assert response.status_code == smc.status_codes.ACCEPTED
        assert response.get_data().decode('utf-8') == 'ping'
