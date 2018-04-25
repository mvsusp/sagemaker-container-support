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
import logging
import os

import sagemaker_containers as smc

logging.basicConfig()
logger = logging.getLogger(__name__)


def keras_framework_training_fn():
    env = smc.Environment.create()

    mod = smc.modules.download_and_import(env.module_dir, env.module_name)

    model = mod.train(**smc.functions.matching_args(mod.train, env))

    if model:
        if hasattr(mod, 'save'):
            mod.save(model, env.model_dir)
        else:
            model_file = os.path.join(env.model_dir, 'saved_model')
            model.save(model_file)


if __name__ == '__main__':
    keras_framework_training_fn()

