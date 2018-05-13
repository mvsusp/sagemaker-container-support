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

import numpy as np
import pytest

from sagemaker_containers import env, functions, modules, mapping
import test

dir_path = os.path.dirname(os.path.realpath(__file__))

USER_MODE_SCRIPT = """
import argparse
import os
import test.fake_ml_framework as fake_ml
import numpy as np

parser = argparse.ArgumentParser()

# Data and model checkpoints directories
parser.add_argument('--training_data_file', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--model_dir', type=str)

args = parser.parse_args()

data = np.load(args.training_data_file)
x_train = data['features']
y_train = data['labels']

model = fake_ml.Model(loss='categorical_crossentropy', optimizer='SGD')

model.fit(x=x_train, y=y_train, epochs=args.epochs, batch_size=args.batch_size)

model_file = os.path.join(args.model_dir, 'saved_model')
model.save(model_file)
"""

USER_SCRIPT = """
import os
import test.fake_ml_framework as fake_ml
import numpy as np

def train(channel_input_dirs, hyperparameters):
    data = np.load(os.path.join(channel_input_dirs['training'], hyperparameters['training_data_file']))
    x_train = data['features']
    y_train = data['labels']

    model = fake_ml.Model(loss='categorical_crossentropy', optimizer='SGD')

    model.fit(x=x_train, y=y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'])

    return model
"""

USER_SCRIPT_WITH_SAVE = """
import os
import test.fake_ml_framework as fake_ml
import numpy as np

def train(channel_input_dirs, hyperparameters):
    data = np.load(os.path.join(channel_input_dirs['training'], hyperparameters['training_data_file']))
    x_train = data['features']
    y_train = data['labels']

    model = fake_ml.Model(loss='categorical_crossentropy', optimizer='SGD')

    model.fit(x=x_train, y=y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'])

    return model

def save(model, model_dir):
    model.save(model_file)
"""


def framework_training_fn():
    training_env = env.TrainingEnv()

    mod = modules.import_module_from_s3(training_env.module_dir, training_env.module_name)

    model = mod.train(**functions.matching_args(mod.train, training_env))

    if model:
        if hasattr(mod, 'save'):
            mod.save(model, training_env.model_dir)
        else:
            model_file = os.path.join(training_env.model_dir, 'saved_model')
            model.save(model_file)


@pytest.mark.parametrize('user_script', [USER_SCRIPT, USER_SCRIPT_WITH_SAVE])
def test_training_framework(user_script):
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', content=user_script))

    hyperparameters = dict(training_data_file='training_data.npz', sagemaker_program='user_script.py',
                           epochs=10, batch_size=64)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    framework_training_fn()

    model_path = os.path.join(env.TrainingEnv().model_dir, 'saved_model')

    model = test.fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.loss == 'categorical_crossentropy'
    assert model.optimizer == 'SGD'


def framework_training_with_script_mode_fn():
    training_env = env.TrainingEnv()

    args = mapping.to_cmd_args(training_env.hyperparameters)

    modules.run_module_from_s3(training_env.module_dir, args, training_env.module_name)

    modules.run(training_env.module_name, args)


@pytest.mark.parametrize('user_script', [USER_MODE_SCRIPT])
def test_script_mode(user_script):
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', content=user_script))

    hyperparameters = dict(training_data_file=os.path.join(channel.path, 'training_data.npz'),
                           sagemaker_program='user_script.py',
                           epochs=10, batch_size=64, model_dir=env.MODEL_PATH)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    framework_training_with_script_mode_fn()

    model_path = os.path.join(env.MODEL_PATH, 'saved_model')

    model = test.fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.loss == 'categorical_crossentropy'
    assert model.optimizer == 'SGD'
