# iRM_Vision_2023

repo for iRM vision.

## Data preparing

We recorded some demo videos to demo auto-aiming capabilities of our robots,
which is located under the folder `./large_data/`.
However, due to the large sizes of the video, it's inappropriate to directly
upload them to GitHub. Hence, to acquire theese sample videos, please download
them at the [UofI box][https://uofi.box.com/s/i6zahotr9id35hurjzy2bq3dcfz0085e].

## Dependencies

Please follow instruction from the RMCV101 repo [here](https://github.com/illini-robomaster/RM_CV_101/blob/master/INSTALL.md).

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

