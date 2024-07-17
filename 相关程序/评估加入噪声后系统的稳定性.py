import numpy as np
from numpy.linalg import inv

# 示例数据（替换为实际数据）
robot_position_data = np.array([[1.2, 3.1], [2.1, 0.8], [2.8, 1.5], [-4.1, 3.5], [1.2, 3.6], [-3.3, 1.2], [-1.5, 0.8], [2.2, 3.2], [-3.2, 0.9], [2.5, 1.5]])  # 各机器人的坐标数据
L_measure_data = np.array([50.94129854803168, 58.65145868607078, 54.67743862373369, 43.44652523368612, 49.60694715261991, 46.77764413681403, 51.86268100501273, 50.679512460729256, 47.259538487338176, 55.27825546524859])  # 测得的声音强度

x_s = 0.5  # 声源的 x 坐标
y_s = -0.2  # 声源的 y 坐标
speaker_position = (x_s, y_s)

# 固定的参数
a = 3.35  # 拟合得到的a值
c = -6.16  # 拟合得到的c值
noise_level = 0.1  # 噪声水平

# 多次实验，统计系统输出的均值和方差
num_experiments = 100
L0_results = []
theta_results = []

for _ in range(num_experiments):
    # 添加高斯噪声
    noisy_robot_positions = robot_position_data + np.random.normal(0, noise_level, robot_position_data.shape)
    noisy_L_measure_data = L_measure_data + np.random.normal(0, noise_level, L_measure_data.shape)

    # 计算机器人到声源的距离
    distance_data = np.linalg.norm(noisy_robot_positions - speaker_position, axis=1)

    # 计算每个机器人相对于声源的角度
    robot_phi_rad_data = np.arctan2(noisy_robot_positions[:, 1], noisy_robot_positions[:, 0])

    # 构建 b_matrix
    b_matrix = noisy_L_measure_data + 20 * np.log10(distance_data) - a - c
    b_matrix_reshaped = b_matrix.reshape(-1, 1)

    # 构建矩阵 A
    A_matrix = np.column_stack((np.ones(len(robot_phi_rad_data)), a * np.cos(robot_phi_rad_data), a * np.sin(robot_phi_rad_data)))

    # 矩阵的求解
    x = inv(A_matrix.T @ A_matrix) @ (A_matrix.T @ b_matrix)

    # 提取 L_0, cos(theta), sin(theta)
    L0 = x[0]
    cos_theta = x[1]
    sin_theta = x[2]

    # 计算 theta
    theta = np.arctan2(sin_theta, cos_theta)
    
    # 保存结果
    L0_results.append(L0)
    theta_results.append(np.rad2deg(theta))


print(L0_results)   
print(theta_results)

# 计算均值和方差
L0_mean = np.mean(L0_results)
L0_std = np.std(L0_results)
theta_mean = np.mean(theta_results)
theta_std = np.std(theta_results)

print(f'L0均值: {L0_mean:.2f}, L0标准差: {L0_std:.2f}')
print(f'theta均值: {theta_mean:.2f} degrees, theta标准差: {theta_std:.2f} degrees')




