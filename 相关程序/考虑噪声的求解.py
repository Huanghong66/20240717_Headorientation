import numpy as np
from numpy.linalg import inv

# 示例数据（替换为实际数据）
robot_position_data = np.array([[1.2, 3.1], [2.1, 0.8], [2.8, 1.5], [-4.1, 3.5], [1.2, 3.6], [-3.3, 1.2], [-1.5, 0.8], [2.2, 3.2], [-3.2, 0.9], [2.5, 1.5]])  # 各机器人的坐标数据
L_measure_data = np.array([50.42674204030757, 58.09098593205243, 54.237346425947734, 43.435189130737925, 49.164035739463785, 46.77448947059422, 51.871405823082384, 50.215870004151455, 47.2451450727298, 54.766850704805435])  # 测得的声音强度

# 假设真实的L0和theta（单位为度）
true_L0 = 65.0
true_theta = -33.0

x_s = 0.5  # 声源的 x 坐标
y_s = -0.2  # 声源的 y 坐标
speaker_position = (x_s, y_s)

# 添加高斯噪声
noise_level = 0.1  # 控制噪声的大小
noisy_robot_positions = robot_position_data + np.random.normal(0, noise_level, robot_position_data.shape)
 
print(f'加入噪声后的机器人位置坐标为:', noisy_robot_positions)


# 计算机器人到声源的距离
distance_data = np.linalg.norm(noisy_robot_positions - speaker_position, axis=1)
print(f'机器人到声源的距离为:', distance_data)

a = 3.35  # 拟合得到的a值
c = -6.16  # 拟合得到的c值

# 计算每个机器人相对于声源的角度
robot_phi_rad_data = np.arctan2(noisy_robot_positions[:, 1] - y_s, noisy_robot_positions[:, 0] - x_s)
print(f'各机器人的角度转换得到的弧度为:', robot_phi_rad_data)


# 构建 b_matrix
b_matrix = L_measure_data + 20 * np.log10(distance_data) - a - c
# 使用reshape函数将其转换为i*1的数组
b_matrix_reshaped = b_matrix.reshape(-1, 1)
print(f'矩阵b为:', b_matrix_reshaped)

# 构建矩阵 A
A_matrix = np.column_stack((np.ones(len(robot_phi_rad_data)), a * np.cos(robot_phi_rad_data), a * np.sin(robot_phi_rad_data)))
print(f'矩阵A为:', A_matrix)

# 矩阵的求解
x = inv(A_matrix.T @ A_matrix) @ (A_matrix.T @ b_matrix)
# 使用reshape函数将其转换为i*1的数组
x_reshaped = x.reshape(-1, 1)
# print(f'解得的x为:', x_reshaped)

# 提取 L_0, cos(theta), sin(theta)
L0 = x[0]
cos_theta = x[1]
sin_theta = x[2]

# 计算 theta
theta = np.arctan2(sin_theta, cos_theta)

L0_relative_error = (L0 - true_L0) / true_L0 * 100
theta_relative_error = (np.rad2deg(theta) - true_theta) / true_theta * 100
print(theta_relative_error)


# 打印结果
print(f'加入噪声的情况下距离发话者1m的声音强度L0预测为: {L0}, L0相对误差: {L0_relative_error:.2f}%')
print(f'加入噪声的情况下发话者的发话角度theta预测为: {np.rad2deg(theta)} degrees, theta相对误差: {theta_relative_error:.2f}%')
