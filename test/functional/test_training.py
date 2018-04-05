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
from sagemaker_containers import Training


@pytest.fixture(name='model_file')
def fixture_model_file(opt_ml):
    return str(opt_ml.join('output/saved_model.json'))


@pytest.fixture(name='opt_ml')
def create_opt_ml(tmpdir):
    environment.base_dir = str(tmpdir)
    tmpdir.mkdir('output')
    tmpdir.mkdir('code')
    return tmpdir


def test_training_from_train_fn(opt_ml, model_file):
    script = "def train(chanel_dirs, hps): return {'trained': True, 'saved': False}"

    write_user_script(opt_ml, script)

    def framework_train_fn(user_module, training_environment):
        model = user_module.train(training_environment.channel_dirs, training_environment.hyperparameters)
        save_model(model, model_file)

    Training.from_train_fn(train_fn=framework_train_fn).start()

    assert load_model(model_file) == {'trained': True, 'saved': True}


def test_training_from_module(opt_ml, model_file):
    script = """
import sagemaker_containers.training_environment as env


def train():
    model = {'trained': True, 'saved': False}

    with open(os.path.join(env.model_dir, 'saved_model.json', 'w')) as f:
        json.dump(model, f)
        
if __name__ == '__main__':
    train()
"""

    write_user_module(opt_ml, script)

    Training.from_module(module_name='customer_module.train').start()

    assert load_model(model_file) == {'trained': True, 'saved': True}


def save_model(model, model_file):
    model['saved'] = True
    with open(model_file, 'w') as f:
        json.dump(model, f)


def load_model(model_file):
    with open(model_file, 'r') as f:
        return json.load(f)


def write_user_script(tmpdir, code):
    user_script_file = tmpdir.join('code/user_script.py')

    user_script_file.write(code)
    return str(user_script_file)


def write_user_module(tmpdir, code):
    user_module = tmpdir.mkdir('customer_module').join('train.py')

    user_module.write(code)
    return str(user_module)
