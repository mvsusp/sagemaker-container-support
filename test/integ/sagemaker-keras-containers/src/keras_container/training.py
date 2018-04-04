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
import os

from sagemaker_containers import TrainingEngine

engine = TrainingEngine()


@engine.train()
def train(user_module, training_environment):
    """ Runs training on a user supplied module.

    Training is invoked by calling a "train" function in the user supplied module.
    """
    model = user_module.train(**training_environment.training_parameters)
    if model:
        if hasattr(user_module, 'save'):
            user_module.save(model, training_environment.model_dir)
        else:
            _default_save(model, training_environment)


def _default_save(model, training_environment):
    """Default logic to save a model to self.model_dir folder (/opt/ml/model).
    This function is called when a customer script does not provide a save() function
        Args:
            model : module to save."""

    model_file = os.path.join(training_environment.model_dir, 'saved_model')
    model.save(model_file)
