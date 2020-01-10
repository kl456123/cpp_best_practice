#!/bin/bash

# generate protoc
protoc --cpp_out=. third_party/protos/*.proto
