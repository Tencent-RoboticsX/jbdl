# xmirror

## Introduction

xmirror is a robot 3D render tool based on MeshCat. It can display robot model on Web by simple Python code.

xmirror support obj, dae, stl 3D model format.

xmirror support urdf and mujuco xml.

## Installation

TODO:



## A example of cart-pole

see in example/cart_pole_test.py

### Import dependence:

```python
from xmirror.robot import RobotModel
import meshcat
import time
import numpy as np
```

### Open visualizer:

```python
vis = meshcat.Visualizer()
vis.open()
sleep_time = 0.5
```

### Load robot model by urdf:

```python
urdf_path = "cartpole.urdf"
robot_model = RobotModel(vis=vis, name="test_bot", id=1, xml_path=urdf_path)
robot_model.render(sleep_time)
```

### Set joint state:

```python
for t in np.arange(0.0, 2, 0.02):
    robot_model.set_joint_state(joint_name="slider_to_cart", state=t)
    time.sleep(0.01)
for t in np.arange(0.0, 1, 0.01):
    robot_model.set_joint_state(joint_name="cart_to_pole", state=t)
    time.sleep(0.01)
for t in np.arange(0.0, 1.01, 0.01):
    robot_model.set_joint_state(joint_name="cart_to_pole", state=1.0 - t)
    time.sleep(0.01)
```

