import numpy as np

robot_position_data = np.array([[1.2, 3.1], [2.1, 0.8], [2.8, 1.5], [-4.1, 3.5], [1.2, 3.6], [-3.3, 1.2], [-1.5, 0.8], [2.2, 3.2], [-3.2, 0.9], [2.5, 1.5]])  # 各机器人的坐标数据
L0 = 65
theta = -33
theta_rad = np.deg2rad(theta)

x_s = 0.5  # 声源的 x 坐标
y_s = -0.2  # 声源的 y 坐标
speaker_position = (x_s, y_s)

a = 3.35  # 拟合得到的a值
c = -6.16  # 拟合得到的c值

# 计算机器人到声源的距离
distance_data = np.linalg.norm(robot_position_data - speaker_position, axis=1)
print(f'机器人到声源的距离为:', distance_data)

# 计算每个机器人相对于声源的角度
robot_phi_rad_data = np.arctan2(robot_position_data[:, 1]  - y_s, robot_position_data[:, 0] - x_s)
print(f'各机器人的角度转换得到的弧度为:', robot_phi_rad_data)

L_calculate_data = L0 - 20 * np.log10(distance_data) + a * (1 + np.cos(robot_phi_rad_data - theta_rad)) + c
L_calculate_data_str = ', '.join(str(value) for value in L_calculate_data)
print(f'各机器人计算得到的声音强度为: {L_calculate_data_str}')
