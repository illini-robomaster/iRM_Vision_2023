### Jetson Aiming

Using pipelined_vision, sending relative yaw and pitch to C板. ...哥们还是用中文吧
在RMUL23 branch的基础上，删去了aimer里的tracker，从detection -> pnp -> tracking 输出absolute yaw/pitch， 改为了detection -> pnp 从pnp中直接输出目标的xyz三维坐标，再转化成relative absolute yaw/pitch发送给下位机。

具体的修改见Aiming/Aim.py line 32起的preprocess()和process_one()

### Gimbal controlling

C板的程序修改很少，程序上传至了iRM_Embedded的PCY/aiming_assistance分支，具体就是在line 260左右的4310控制时，会根据jetson传来的relative angles给电机转动速度一个修正（yaw/pitch 乘一个constant加到转速上）

### Notes and TODOs

1.命名问题：这套程序是基于RMUC23修改的，为了防止奇怪的bug，部分变量名未作修改，所以尽管通讯时，jetson传到下位机的yaw和pitch叫做abs_yaw,abs_pitch, 但实际上在把preprocess里的gimbal_yaw,即current yaw hardcode成0之后，现在pnp解算出来传给下位机的实际上不是absolute yaw而是relative yaw

2.TODO(1)整套程序由于时间和突发的硬件问题未能上机测试，目前是参数和下位机程序肯定是有问题的，需要后期调整。

3.TODO（2）根据上述测试情况酌情添加:为了防止和鼠标/操控手抢权限，可在下位机程序中为鼠标加上死区，在鼠标运动幅度较小时不启用辅助瞄准