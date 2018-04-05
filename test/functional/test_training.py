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
import json

import pytest
import sagemaker_containers.environment as environment
from sagemaker_containers import TrainingEngine, App


@pytest.fixture(name='model_file')
def fixture_model_file(tmpdir):
    return str(tmpdir.mkdir('output').join('saved_model.json'))


@pytest.fixture(name='create_user_module')
def fixture_create_user_module(tmpdir):
    environment.base_dir = str(tmpdir)
    user_script_file = tmpdir.mkdir('code').join('user_script.py')

    script = "def train(chanel_dirs, hps): return {'trained': True, 'saved': False}"

    user_script_file.write(script)


def test_register_training_with_decorator(create_user_module, model_file):
    engine = TrainingEngine()

    @engine.framework_train_fn()
    def framework_train(user_module, training_environment):
        model = user_module.train(training_environment.channel_dirs, training_environment.hyperparameters)
        save_model(model, model_file)

    app = App()
    app.register_engine(engine)

    app.run()

    assert load_model(model_file) == {'trained': True, 'saved': True}


def test_register_training_with_fn(create_user_module, model_file):
    def framework_train(user_module, training_environment):
        model = user_module.train(training_environment.channel_dirs, training_environment.hyperparameters)
        save_model(model, model_file)

    engine = TrainingEngine(framework_train)

    app = App()
    app.register_engine(engine)

    app.run()

    assert load_model(model_file) == {'trained': True, 'saved': True}


def test_app_run_with_decorator(create_user_module, model_file):
    app = App()

    @app.training_engine.framework_train_fn()
    def framework_train(user_module, training_environment):
        model = user_module.train(training_environment.channel_dirs, training_environment.hyperparameters)
        save_model(model, model_file)

    app.run()

    assert load_model(model_file) == {'trained': True, 'saved': True}


def save_model(model, model_file):
    model['saved'] = True
    with open(model_file, 'w') as f:
        json.dump(model, f)


def load_model(model_file):
    with open(model_file, 'r') as f:
        return json.load(f)
