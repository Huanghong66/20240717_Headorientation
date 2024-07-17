import numpy as np
from numpy.linalg import inv

# 假设已有数据存储在变量中 以下变量均为已知
# robot_phi_degree_data: 各机器人的角度（单位为度）
# L_measure_data: 各机器人的测量声音强度
# x_s, y_s: 声源的世界坐标
# a: 角度衰减系数

# 示例数据（替换为实际数据）
robot_position_data = np.array([[1.2, 3.1], [2.1, 0.8], [2.8, 1.5], [-4.1, 3.5], [1.2, 3.6], [-3.3, 1.2], [-1.5, 0.8], [2.2, 3.2], [-3.2, 0.9], [2.5, 1.5]])  # 各机器人的坐标数据
L_measure_data = np.array([50.94129854803168, 58.65145868607078, 54.67743862373369, 43.44652523368612, 49.60694715261991, 46.77764413681403, 51.86268100501273, 50.679512460729256, 47.259538487338176, 55.27825546524859])  # 测得的声音强度

x_s = 0.5  # 声源的 x 坐标
y_s = -0.2  # 声源的 y 坐标
speaker_position = (x_s, y_s)
# 计算机器人到声源的距离
distance_data = np.linalg.norm(robot_position_data - speaker_position, axis=1)
print(f'机器人到声源的距离为:', distance_data)

a = 3.35  # 拟合得到的a值
c = -6.16  # 拟合得到的c值

# 计算每个机器人相对于声源的角度
robot_phi_rad_data = np.arctan2(robot_position_data[:, 1] - y_s, robot_position_data[:, 0] - x_s)
print(f'各机器人的角度转换得到的弧度为:', robot_phi_rad_data)

# 构建 b_matrix
b_matrix = L_measure_data + 20 * np.log10(distance_data) - a - c
# 使用reshape函数将其转换为i*1的数组
b_matrix_reshaped = b_matrix.reshape(-1, 1)
print(f'矩阵b为:', b_matrix_reshaped)

# 构建矩阵 A
A_matrix = np.column_stack((np.ones(len(robot_phi_rad_data)), a * np.cos(robot_phi_rad_data), a * np.sin(robot_phi_rad_data)))
print(f'矩阵A为:', A_matrix)

#  矩阵的求解
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

# 打印结果
print(f'距离发话者1m的声音强度L0预测为: {L0:.2f}')
print(f'发话者的发话角度theta预测为: {np.rad2deg(theta):.2f} degrees')


