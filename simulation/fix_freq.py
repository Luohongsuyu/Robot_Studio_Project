import pybullet as p
import os
import pybullet_data
import numpy as np
from math import sin, pi
import random
from deap import base, creator, tools, algorithms
import time

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

        # Fixed frequency for all joints
        fixed_frequency = 1.5

        # Simulate for a set time
        for i in range(int(simulation_duration / time_step)):
            current_time = i * time_step

            # Skip movement for the first 0.5 seconds
            if current_time < 0.5:
                p.stepSimulation()  # Just step the simulation without any movement
                continue

            for joint_index in range(8):  # Apply to 8 joints
                amplitude, phase = action[joint_index]
                # Compute the target position using sine wave: amplitude * sin(2 * pi * fixed_frequency * t + phase)
                target_position = amplitude * sin(2 * pi * fixed_frequency * current_time + phase)
                p.setJointMotorControl2(self.robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
        
            p.stepSimulation()  # Step the simulation forward
        # Get new position of the robot's base after simulation
        current_position, _ = p.getBasePositionAndOrientation(self.robotId)

        # Reward: distance moved along x-axis
        reward = current_position[0]
        return reward

    def close(self):
        p.disconnect()


# 定义适应度函数，包含对超过振幅限制的个体的惩罚
def fitness_function(individual):
    env = SineWaveRobotEnv()
    env.reset()

    # 将个体的基因转换为动作格式，每 3 个参数表示一个关节的 (振幅, 频率, 相位)
    action = np.reshape(individual, (8, 2))  # 4 joints, 3 parameters each
    
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

    # 基因的范围是 0.1 到 0.349066 (amplitude), 0 到 2*pi (phase)
    toolbox.register("attr_float_amp", random.uniform, 0.1, 0.349066)
    toolbox.register("attr_float_phase", random.uniform, 0, 2 * pi)
    
    # 定义每个个体的生成方式：8 joints * 2 parameters (amplitude, phase) = 16 genes
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_float_amp, toolbox.attr_float_phase), n=8)
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
    camera_distance = 1.5  # 摄像机距离
    camera_yaw = 0  # 从 y 轴方向观察
    camera_pitch = -30  # 设置俯仰角
    camera_target_position = [0, 0, 0.2]  # 摄像机的目标位置
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, 
                                 cameraPitch=camera_pitch, cameraTargetPosition=camera_target_position)

    # Get the number of joints
    num_joints = p.getNumJoints(robotId)
    
    # 将最佳个体的基因转换为动作格式
    joint_params = np.reshape(best_solution, (8, 2))  # 将最佳个体的基因转换为 (8 joints, 2 parameters)

    # Simulate for a few seconds with joint movement
    simulation_duration = 10 # seconds
    time_step = 1. / 240.  # Simulation time step

    # Fixed frequency
    fixed_frequency = 1.5

    # Simulate for a set time
    for i in range(int(simulation_duration / time_step)):
        current_time = i * time_step

        # Skip movement for the first 0.5 seconds
        if current_time < 0.5:
            p.stepSimulation()  # Just step the simulation without any movement
            time.sleep(time_step)
            continue

        for joint_index in range(min(num_joints, 8)):  # Apply to 8 joints
            amplitude, phase = joint_params[joint_index]
            # Compute the target position using sine wave: amplitude * sin(2 * pi * fixed_frequency * t + phase)
            target_position = amplitude * sin(2 * pi * fixed_frequency * current_time + phase)
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
    
        p.stepSimulation()  # Step the simulation forward
        time.sleep(time_step)  # Control the real-time speed of the simulation

    # Get new position of the robot's base after simulation
    current_position, _ = p.getBasePositionAndOrientation(robotId)

    # Reward: distance moved along x-axis
    reward = current_position[0]
    # Disconnect the simulation after visualization
    p.disconnect()
    return reward





# 运行遗传算法
#best_solution = genetic_algorithm()


#good_unlimit_x = [0.47960052940502174, 5.9124027115146705, 0.5174492431223019, 3.59110361921881, 0.41211311089730984, 3.575135033774271, 0.20530799875426714, 0.161666856038285, 0.8273818611739109, 4.1906441437252635, 0.015186542531014702, 0.8984790937197128, 0.40100814935002094, 6.038468988092496, 0.2244445068836761, 2.7781517128054816]
# 输入每个电机的振幅和相位
test_solution = [0.3369560724332124, 1.6334908368186318, 0.3476805068159392, 4.444848536852759, 0.2826176689523937, 4.930316131073677, 0.3430500918795555, 1.5678051492194085, 0.34863203867587855, 5.2583656785273485, 0.33877365624509215, 2.827034620904344, 0.3479262733413843, 1.8460141729365889, 0.34837665675548135, 5.288566699073646]
test_solution_insimi = [0.3369560724332124, 2.9334908368186318, 0.3476805068159392, 4.444848536852759, 0.2826176689523937, 2.930316131073677, 0.3430500918795555, 1.5678051492194085, 0.34863203867587855, 5.2583656785273485, 0.33877365624509215, 2.827034620904344, 0.3479262733413843, 1.8460141729365889, 0.34837665675548135, 5.288566699073646]

a = simulate_best_solution(test_solution)
#b = fitness_function(test_solution)
#print(a)
#print(b)
#print_joint_info(test_solution)