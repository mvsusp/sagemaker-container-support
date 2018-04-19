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

import json

from flask import Flask, Response, request

import sagemaker_container_support as scs


def default_error_handler(exception):
    """ Default error handler. Returns 500 status with no content.

    :param exception: the exception that triggered the error
    :return: 500 response
    """

    log.error(exception)
    return '', 500


def healthcheck():
    """Default healthcheck handler. Returns 200 status with no content. Note that the
    `InvokeEndpoint API`_ contract requires that the service only returns 200 when
    it is ready to start serving requests.

    :return: 200 response if the serer is ready to handle requests.
    """
    return '', 200


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
OCTET_STREAM_CONTENT_TYPE = "application/octet-stream"
ANY_CONTENT_TYPE = '*/*'
UTF8_CONTENT_TYPES = [JSON_CONTENT_TYPE, CSV_CONTENT_TYPE]


def handle_invoke_exception(e):
    data = json.dumps(e.message)
    if isinstance(e, UnsupportedContentTypeError):
        # Unsupported Media Type
        return 415, data
    elif isinstance(e, UnsupportedAcceptTypeError):
        # Not Acceptable
        return 406, data
    elif isinstance(e, UnsupportedInputShapeError):
        # Precondition Failed
        return 412, data
    else:
        log.exception(e)
        raise e


def run(transformer):
    """Creates and executes the Flask app
    Args:
        transformer (any instance of class that implements initialize() and transform())
    """
    env = scs.Environment.create()

    transformer.initialize(env)

    app = Flask(env.module_name)
    app.add_url_rule('/ping', 'healthcheck', healthcheck)

    def invoke():
        try:
            response_data, output_content_type = transformer.transform(RequestEnvironment.create(env))
            # OK
            ret_status = 200
        except Exception as e:
            ret_status, response_data = handle_invoke_exception(e)
            output_content_type = JSON_CONTENT_TYPE

        return Response(response=response_data,
                        status=ret_status,
                        mimetype=output_content_type)

    app.add_url_rule('/invocations', 'invoke', invoke, methods=["POST"])
    app.register_error_handler(Exception, default_error_handler)

    return app


def get_content_type():
    return request.headers.get('ContentType', request.headers.get('Content-Type', JSON_CONTENT_TYPE))


def get_accept():
    return request.headers.get('Accept', JSON_CONTENT_TYPE)


def get_request_data(content_type):
    # utf-8 decoding is automatic in Flask if the Content-Type is valid. But that does not happens always.
    is_utf8_content_type = content_type in UTF8_CONTENT_TYPES
    return request.get_data().decode('utf-8') if is_utf8_content_type else request.get_data()


class Request(object):
    def __init__(self, content_type, accept, data):
        self._data = data
        self._accept = accept
        self._content_type = content_type

    @classmethod
    def create(cls):
        content_type = get_content_type()
        return cls(content_type=content_type,
                   accept=get_accept(),
                   data=get_request_data(content_type))

    @property
    def content_type(self):
        return self._content_type

    @property
    def accept(self):
        return self._accept

    @property
    def data(self):
        return self._data

    @property
    def user_module(self):
        return self._user_module
