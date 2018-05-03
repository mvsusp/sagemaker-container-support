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

import sagemaker_containers.collections
import sagemaker_containers.content_types
import sagemaker_containers.environment
import sagemaker_containers.functions
import sagemaker_containers.modules
import sagemaker_containers.server
import sagemaker_containers.status_codes
import sagemaker_containers.worker  # noqa ignore=F401

from sagemaker_containers.environment import Environment, ServingEnvironment, TrainingEnvironment  # noqa ignore=F401
#  imported but unused
