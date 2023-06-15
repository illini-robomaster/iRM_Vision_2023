# iRM_Vision_2023

repo for iRM vision.

## Contributing

Now the repo uses CI to check code style. We decided pylint is too painful to use.
Instead we use `pycodestyle` and `pydocstyle` to check if the code conform to autopep8
and doc standards.

Code that fails the CI will **NOT** be merged.

To check if your code is conform to the standard, run

```bash
pycodestyle --max-line-length=100 .
pydocstyle .
```

These two tools can be installed from either pip or anaconda.

The CI will also tell if your code is failing, but please don't rely on repeatedly submitting
commits to check your coding style.

To automatically fix your code to autopep8 standard, you can run

```bash
autopep8 --in-place --aggressive --aggressive --max-line-length=100 --exclude="mvsdk.py" --recursive .
```

## Data preparing

We recorded some demo videos to demo auto-aiming capabilities of our robots,
which is located under the folder `./large_data/`.
However, due to the large sizes of the video, it's inappropriate to directly
upload them to GitHub. Hence, to acquire theese sample videos, please download
them at the [UofI box](https://uofi.box.com/s/i6zahotr9id35hurjzy2bq3dcfz0085e).

## Dependencies

Please follow instruction from the RMCV101 repo [here](https://github.com/illini-robomaster/RM_CV_101/blob/master/INSTALL.md).

## TODOs

- Implement a more robust tracker (e.g., EKF / Kalman)
- Implement depth estimators (either use D455 or estimate from monocular cameras)
- Optimize code efficiency on Jetson
- Update building instruction for getting video files / MDVS SDK

## File Structure

```
- Aiming/              --> code needed aiming target armor
    - tracking/        --> Keep code for tracking here
- Camera/              --> Camera utils
- Communication/       --> Keep code for communication to chassis here
- Detection/           --> Yolo code
- Utils/               --> misc.
- vision.py            --> vision driver
- config.py            --> global config (parameters and class instance)
```

## CHANGLELOG

2023-02-14 v0.0.1 release for 2023 season midterm. See notes [here](https://github.com/illini-robomaster/iRM_Vision_2023/pull/1).
