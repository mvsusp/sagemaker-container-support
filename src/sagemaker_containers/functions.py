# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import inspect

import six

import sagemaker_containers as smc


def signature(fn):
    if six.PY2:
        # noinspection PyDeprecation
        arg_spec = inspect.getargspec(fn)
        return arg_spec.args, arg_spec.varargs, arg_spec.keywords

    sig = inspect.signature(fn)

    def filter_parameters(kind):
        return [
            p.name for p in sig.parameters.values()
            if p.kind == kind
        ]

    args = filter_parameters(inspect.Parameter.POSITIONAL_OR_KEYWORD)

    vargs = filter_parameters(inspect.Parameter.VAR_POSITIONAL)

    kwargs = filter_parameters(inspect.Parameter.VAR_KEYWORD)

    return (args,
            vargs[0] if vargs else None,
            kwargs[0] if kwargs else None)


def matching_args(fn, environment):
    args, _, kwargs = signature(fn)

    if kwargs:
        return environment

    return smc.collections.split_by_criteria(environment, set(args))[0]
