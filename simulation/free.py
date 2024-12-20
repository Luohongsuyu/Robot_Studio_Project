import pybullet as p
import os
import pybullet_data
import numpy as np
from math import sin, pi
import random
import time
from deap import base, creator, tools, algorithms

# 环境定义
class SineWaveRobotEnv:
    def __init__(self):
        # Connect to PyBullet
        self.physicsClient = p.connect(p.DIRECT)  # Use DIRECT for headless (no GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        
        # Load the robot URDF
        urdf_model = "spider.SLDASM.urdf"
        orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
        self.robotId = p.loadURDF(urdf_model, [0, 0, 0.12], orientation)
        
        p.setGravity(0, 0, -9.8)
        
        # Get the initial base position
        self.prev_position, _ = p.getBasePositionAndOrientation(self.robotId)
        
        return np.array(self.prev_position)

    def step(self, action):
        # Simulate for a few seconds with joint movement
        simulation_duration = 10  # seconds
        time_step = 1. / 240.  # Simulation time step

        # Simulate for a set time
        for i in range(int(simulation_duration / time_step)):
            current_time = i * time_step

            # Skip movement for the first 0.5 seconds
            if current_time < 0.5:
                p.stepSimulation()  # Just step the simulation without any movement
                continue


            for joint_index in range(8):  # Apply to 8 joints
                amplitude, frequency, phase = action[joint_index]
                # Compute the target position using sine wave: amplitude * sin(2 * pi * frequency * t + phase)
                target_position = amplitude * sin(2 * pi * frequency * current_time + phase)
                p.setJointMotorControl2(self.robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
        
            p.stepSimulation()  # Step the simulation forward
        # Get new position of the robot's base after simulation
        current_position, _ = p.getBasePositionAndOrientation(self.robotId)

        # Reward: distance moved along x-axis
        reward = current_position[0] #- self.prev_position[0]
        return reward

    def close(self):
        p.disconnect()

# 定义适应度函数，包含对超过振幅限制的个体的惩罚
def fitness_function(individual):
    env = SineWaveRobotEnv()
    env.reset()

    # 将个体的基因转换为动作格式，每 3 个参数表示一个关节的 (振幅, 频率, 相位)
    action = np.reshape(individual, (8, 3))  # 4 joints, 3 parameters each
    
    # 惩罚因子，如果有振幅超过 20 度（0.349066 弧度），将适应度乘以这个因子
    penalty_factor = 1.0
    max_amplitude = 0.349066  # 20度的弧度值

    # 遍历每个关节的振幅，检查是否超出限制
    for joint_params in action:
        amplitude = joint_params[0]  # 振幅
        if amplitude > max_amplitude:
            penalty_factor *= 0.05  # 给予严重惩罚，将适应度乘以 0.1

    # 计算适应度值（假设是机器人移动的距离）
    reward = env.step(action)
    
    env.close()

    # 应用惩罚因子，如果振幅超标则显著降低适应度
    return reward * penalty_factor,

# 使用 DEAP 构建遗传算法
def genetic_algorithm():
    # 创建适应度和个体
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 初始化工具箱
    toolbox = base.Toolbox()

    # 基因的范围是 0.1 到 0.5 (振幅), 0.5 到 2.0 (频率), 0 到 2*pi (相位)
    toolbox.register("attr_float", random.uniform, 0.1, 0.349066)  # 振幅的范围限制为 0.1 到 0.349066 弧度
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=24)  # 8 joints * 3 parameters = 6
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 创建统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 遗传算法参数
    population_size = 50
    generations = 1000
    crossover_rate = 0.7
    mutation_rate = 0.15

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 使用算法库的遗传算法并记录每一代的统计信息
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate, 
                                              ngen=generations, stats=stats, verbose=True)

    # 获取最佳个体
    best_individual = tools.selBest(population, 1)[0]
    print("Best individual is: ", best_individual)
    print("Best fitness is: ", best_individual.fitness.values[0])



# 使用最佳个体做10秒钟的模拟，并渲染动画
def simulate_best_solution(best_solution):
    # Start the PyBullet physics simulation engine
    physicsClient = p.connect(p.GUI)  # Use GUI to visualize the simulation

    # Set the current directory as the search path for URDF files
    current_dir = os.getcwd()
    p.setAdditionalSearchPath(current_dir)  # Ensure PyBullet can find the URDF file
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load the URDF model from the current directory
    urdf_model = "spider.SLDASM.urdf"
    orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])  # Convert Euler angles to quaternion
    robotId = p.loadURDF(urdf_model, [0, 0, 0.12], orientation)

    # Set gravity in the environment
    p.setGravity(0, 0, -9.8)


    # 设置摄像机视角从 y 轴方向观察
    camera_distance = 1.0  # 摄像机距离
    camera_yaw = 0  # 从 y 轴方向观察
    camera_pitch = -30  # 设置俯仰角
    camera_target_position = [0, 0, 0.2]  # 摄像机的目标位置
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, 
                                 cameraPitch=camera_pitch, cameraTargetPosition=camera_target_position)

    # Get the number of joints
    num_joints = p.getNumJoints(robotId)
    
    # 将最佳个体的基因转换为动作格式
    joint_params = np.reshape(best_solution, (8, 3))  # 将最佳个体的基因转换为 (8 joints, 3 parameters)

    # Simulate for a few seconds with joint movement
    simulation_duration = 20  # seconds
    time_step = 1. / 240.  # Simulation time step

    # Simulate for a set time
    for i in range(int(simulation_duration / time_step)):
        current_time = i * time_step

        # Skip movement for the first 0.5 seconds
        if current_time < 0.5:
            p.stepSimulation()  # Just step the simulation without any movement
            time.sleep(time_step)
            continue


        for joint_index in range(min(num_joints, 8)):  # Apply to 8 joints
            amplitude, frequency, phase = joint_params[joint_index]
            # Compute the target position using sine wave: amplitude * sin(2 * pi * frequency * t + phase)
            target_position = amplitude * sin(2 * pi * frequency * current_time + phase)
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
    
        p.stepSimulation()  # Step the simulation forward
        time.sleep(time_step)  # Control the real-time speed of the simulation
    # Get new position of the robot's base after simulation
    current_position, _ = p.getBasePositionAndOrientation(robotId)

    # Reward: distance moved along x-axis
    reward = current_position[0] #- self.prev_position[0]
    # Disconnect the simulation after visualization
    p.disconnect()
    return reward


def print_joint_info(action):
    # Start the PyBullet physics simulation engine
    physicsClient = p.connect(p.DIRECT)  # Use DIRECT for headless (no GUI)

    # Set the current directory as the search path for URDF files
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load the URDF model from the current directory
    urdf_model = "spider.SLDASM.urdf"
    orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
    robotId = p.loadURDF(urdf_model, [0, 0, 0.12], orientation)

    # Get the number of joints
    num_joints = p.getNumJoints(robotId)

    # 将动作转换为关节控制参数格式 (每个关节有 3 个参数: 振幅, 频率, 相位)
    joint_params = np.reshape(action, (num_joints, 3))

    # 打印每个关节的名字和对应的振幅、频率和相位
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robotId, joint_index)
        joint_name = joint_info[1].decode('utf-8')  # 获取关节名字
        amplitude, frequency, phase = joint_params[joint_index]
        
        # 打印结果
        print(f"Joint {joint_index} ({joint_name}): Amplitude = {amplitude}, Frequency = {frequency}, Phase = {phase}")

    # 断开仿真连接
    p.disconnect()




# 运行遗传算法
#best_solution = genetic_algorithm()

test_solution = [0.15815873812652034, 0.2606073141570008, 0.26760547795818207, 0.336063872418118, 0.5674556101739952, 0.3310464182295256, 0.22956392089497743, 0.3497133585272746, 0.11982812268261982, 0.3426817131911689, 0.28420028456162816, 0.17913647798836052, 0.20341510778812516, 0.22897667970717375, 0.2997119944758957, 0.27714525384087063, 0.2601073937127786, 0.09089474918784778, 0.3465003107322865, 0.2906427640405874, 0.09141995680400822, 0.3274777494858092, 0.5962394562025305, 0.20905920278007215]
pareto = [0.3013224647775507, 1.1326591136032793, 1.2083176269984182, 0.33861962784851485, 1.0406788307583756, 0.9769113855016969, -1.0874752722867103, 1.1272630516688236, 0.7418284020572621, 0.3118873032161723, 1.0865923569644773, 2.58137053019148, 0.3448190721828047, 1.0692905555031713, 0.9868322483872967, 0.28555407498730456, 1.1360906090643743, 2.2884478425953056, -1.0795589899396796, 1.1215223519909183, 4.906374874776356, 0.34698214007113926, 1.1158231558413643, 5.007879369523769]
a = simulate_best_solution(pareto)
#b = fitness_function(test_solution)
print(a)
#print(b)
#print_joint_info(test_solution)