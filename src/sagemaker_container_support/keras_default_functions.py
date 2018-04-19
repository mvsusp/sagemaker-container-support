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

from keras import Model

import sagemaker_container_support as scs
from sagemaker_container_support.transformer import Transformer


def default_model_fn(model_dir):
    return Model.load_weights(model_dir)


def default_input_fn(data, content_type):
    return scs.serializers.loads(data, content_type)


def default_predict_fn(model, input_data):
    return model(input_data)


def default_output_fn(prediction, accept):
    return scs.serializers.dumps(prediction, accept)


if __name__ == '__main__':
    env = scs.Environment.create()
    user_module = scs.modules.download_and_import(env.module_dir, env.module_name)
    transformer = Transformer.from_functions(user_module, default_model_fn, default_input_fn,
                                             default_predict_fn, default_output_fn)

    scs.worker.run(transformer)