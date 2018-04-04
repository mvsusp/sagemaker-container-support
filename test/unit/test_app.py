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

import pytest
from mock import Mock, patch

from sagemaker_containers import App, TrainingEngine


@pytest.fixture(name='app')
def fixture_app(training_engine):
    app = App()
    app.training_engine = training_engine
    return app


@pytest.fixture(name='training_engine')
def fixture_training_engine():
    return Mock(spec=TrainingEngine())


def test_register(training_engine):
    app = App()

    app.register_engine(training_engine)

    assert app.training_engine == training_engine


def test_register_invalid_engine(app):
    with pytest.raises(ValueError) as error:
        app.register_engine(object())

    assert "'object'> is not a valid engine type" in str(error.value)


@patch('sys.argv', ['fake-program', 'train'])
def test_run(app):
    app.run()

    assert app.training_engine.run.was_called


@patch('sys.argv', ['fake-program', 'serve'])
def test_run_invalid_args(app):
    with pytest.raises(ValueError) as error:
        app.run()

    assert str(error.value) == "Illegal arguments: ['serve']"


@patch('sys.argv', ['fake-program'])
def test_run_without_args(app):
    with pytest.raises(ValueError) as error:
        app.run()

    assert str(error.value) == "Missing argument: train"
