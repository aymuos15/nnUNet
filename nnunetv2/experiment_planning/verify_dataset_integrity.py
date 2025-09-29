#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Import from new modular structure
from nnunetv2.experiment_planning.dataset.validation.integrity_checker import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join


# All validation functions are now available from the new modular structure
# This file serves as a backward-compatible entry point


if __name__ == "__main__":
    # investigate geometry issues
    example_folder = join(nnUNet_raw, 'Dataset250_COMPUTING_it0')
    num_processes = 6
    verify_dataset_integrity(example_folder, num_processes)
