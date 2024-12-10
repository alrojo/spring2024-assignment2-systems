#!/bin/bash
pytest -k test_rmsnorm_forward_pass_pytorch
pytest -k test_rmsnorm_forward_pass_triton
