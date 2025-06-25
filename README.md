# Run instructions
To test the face detector, run `python get_validation.py <image_path>`.

To run the validation, run `python get_validation.py validate <reduce>`. The second argument is optional with `True` (default, with reducing image size) or `False` (without reducing image size).

To get the statistics from the validation, `python get_validation.py <validation output file path>`.

To plot the runtime of the face detection for different images, run `python check_runtime.py <validation output file path>`
