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

from mock import ANY, Mock, patch, PropertyMock
import pytest
from six.moves import range

import sagemaker_containers as smc


class MockTransformer(Mock):
    def transform(self):
        return smc.worker.TransformSpec('fake data', smc.content_types.APPLICATION_JSON)


def test_default_ping_fn():
    assert smc.worker.default_healthcheck_fn().status_code == smc.status_codes.OK


@pytest.fixture(name='flask')
def patch_flask():
    property_mock = PropertyMock(return_value='user_program')
    with patch('sagemaker_containers.worker.Flask') as flask, \
            patch('sagemaker_containers.environment.ServingEnvironment.module_name',
                  property_mock):
        yield flask


@pytest.mark.parametrize('module_name, expected_name', [('test_module', 'test_module'), (None, 'user_program')])
def test_run(flask, module_name, expected_name):
    transformer = MockTransformer()
    app = smc.worker.run(transformer, module_name=module_name)

    flask.assert_called_with(import_name=expected_name)

    transformer.initialize.assert_called()

    rules = app.add_url_rule
    rules.assert_any_call(rule='/invocations', endpoint='invocations', view_func=ANY, methods=ANY)

    rules.assert_called_with(rule='/ping', endpoint='ping', view_func=smc.worker.default_healthcheck_fn)

    assert rules.call_count == 2


def test_invocations():
    transformer = MockTransformer()
    app = smc.worker.run(transformer, module_name='test_module')

    with app.test_client() as worker:
        for _ in range(9):
            response = worker.post('/invocations')
            assert response.status_code == smc.status_codes.OK
            assert response.get_data().decode('utf-8') == 'fake data'
            assert response.mimetype == smc.content_types.APPLICATION_JSON
