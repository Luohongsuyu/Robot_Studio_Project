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
                amplitude, frequency, phase = action[joint_index]
                # Compute the target position using sine wave: amplitude * sin(2 * pi * fixed_frequency * t + phase)
                target_position = amplitude * sin(2 * pi * frequency * current_time + phase)
                p.setJointMotorControl2(self.robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
        
            p.stepSimulation()  # Step the simulation forward
        # Get new position of the robot's base after simulation
        current_position, _ = p.getBasePositionAndOrientation(self.robotId)

        # Objective values: maximize x-axis movement, minimize y-axis movement
        x_displacement = current_position[0]
        y_displacement = abs(current_position[1])  # Take absolute value for y-axis displacement
        return x_displacement, y_displacement

    def close(self):
        p.disconnect()

# 定义适应度函数
def fitness_function(individual):
    env = SineWaveRobotEnv()
    env.reset()

    # 将个体的基因转换为动作格式
    action = np.reshape(individual, (8, 3))  # 8 joints, 2 parameters each
    
    # 惩罚因子，如果有振幅超过 20 度（0.349066 弧度），将适应度乘以这个因子
    penalty_factor = 1.0
    max_amplitude = 0.349066  # 20度的弧度值

    # 检查是否超出限制
    for joint_params in action:
        amplitude = joint_params[0]
        if amplitude > max_amplitude:
            penalty_factor *= 0.05  # 严重惩罚

    # 获取两个目标值（x轴和y轴位移）
    x_displacement, y_displacement = env.step(action)
    env.close()

    # 应用惩罚因子，调整两个目标值
    return x_displacement * penalty_factor, y_displacement/penalty_factor

# 使用 DEAP 构建NSGA-II算法
def genetic_algorithm():
    # 创建多目标适应度和个体
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # 最大化x轴位移，最小化y轴位移
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # 初始化工具箱
    toolbox = base.Toolbox()

    # 定义基因范围
    toolbox.register("attr_float_amp", random.uniform, 0.1, 0.349066)
    toolbox.register("attr_float_phase", random.uniform, 0, 2 * pi)
    toolbox.register("attr_float_freq", random.uniform, 0.1, 1.5)
    
    # 定义个体生成方式
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_float_amp, toolbox.attr_float_freq, toolbox.attr_float_phase), n=8)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)  # NSGA-II 选择机制

    # 创建统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 参数设置
    population_size = 50
    generations = 1000
    crossover_rate = 0.7
    mutation_rate = 0.15

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 使用 NSGA-II 算法进行多目标优化
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, 
                              cxpb=crossover_rate, mutpb=mutation_rate, ngen=generations, stats=stats, 
                              verbose=True)

    # 获取Pareto前沿的解集
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)
    best_individual = pareto_front[0][0]
    print("Best individual in Pareto front: ", best_individual)
    print("Fitness (x-displacement, y-displacement): ", best_individual.fitness.values)


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
    joint_params = np.reshape(best_solution, (8, 3))  # 将最佳个体的基因转换为 (8 joints, 2 parameters)

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
            amplitude, frequency, phase = joint_params[joint_index]
            # Compute the target position using sine wave: amplitude * sin(2 * pi * fixed_frequency * t + phase)
            target_position = amplitude * sin(2 * pi * frequency * current_time + phase)
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
best_solution = genetic_algorithm()

# 输入每个电机的振幅和相位
test_solution = [0.3011962397401819, 1.222697253290161, 4.334582333621904, 0.16841364215663907, 0.8626920151726509, 5.704334195974859, 0.1721162259361027, 1.078511242837807, 5.617685243910988, 0.2543030207205102, 1.0086405916967756, 1.344921033831464, 0.23128577527896613, 1.1817760822921395, 4.115073417172827, 0.12576430258104737, 1.0005073390436965, 3.9978968060008775, 0.3440981478155254, 0.9754242695396795, 3.920884610511522, 0.18402485324418752, 0.682586680218157, 2.997252016928269]

#a = simulate_best_solution(test_solution)
#b = fitness_function(test_solution)
#print(a)
#print(b)
#print_joint_info(test_solution)