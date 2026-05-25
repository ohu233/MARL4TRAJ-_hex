import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,          # 基础字体大小
    'axes.labelsize': 20,     # 坐标轴标签字体大小
    'axes.titlesize': 24,     # 标题字体大小
    'xtick.labelsize': 18,    # x轴刻度字体大小
    'ytick.labelsize': 18,    # y轴刻度字体大小
    'legend.fontsize': 16,    # 图例字体大小
    'figure.titlesize': 20,   # 图表标题字体大小
})

# 读取数据
data = pd.read_csv('SAC_5000_eps_inrealmap_44_training_results.csv')

# 计算滑动统计量（窗口大小 50）
window_size = 100

# 奖励统计
data['Moving Average Reward'] = data['Return'].rolling(window=window_size, min_periods=1).mean()
data['Reward Std Dev'] = data['Return'].rolling(window=window_size, min_periods=1).std()

# 损失统计
data['Moving Average Critic Loss'] = data['Critic Loss'].rolling(window=window_size, min_periods=1).mean()
data['Critic Loss Std Dev'] = data['Critic Loss'].rolling(window=window_size, min_periods=1).std()
data['Moving Average Actor Loss'] = data['Actor Loss'].rolling(window=window_size, min_periods=1).mean()
data['Actor Loss Std Dev'] = data['Actor Loss'].rolling(window=window_size, min_periods=1).std()

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 增大图表尺寸以适应更大字体

# 设置网格线
ax1.grid(True, linestyle='--', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.6)

# 绘制奖励图（左）
ax1.plot(data['Episode'], data['Return'], color='lightblue', alpha=0.3, label='Reward')
ax1.plot(data['Episode'], data['Moving Average Reward'], color='red', label='Moving Average Reward')
ax1.fill_between(
    data['Episode'],
    data['Moving Average Reward'] - data['Reward Std Dev'],
    data['Moving Average Reward'] + data['Reward Std Dev'],
    color='blue',
    alpha=0.2,
    label='Reward ± Std Dev'
)
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward')
ax1.set_title('Reward of the path recovery agent without Conv')
ax1.legend()

# 绘制损失图（右）
# 原始损失值
line_critic, = ax2.plot(data['Episode'], data['Critic Loss'], color='orange', alpha=0.3, label='Critic Loss')
line_actor, = ax2.plot(data['Episode'], data['Actor Loss'], color='green', alpha=0.3, label='Actor Loss')

# 平均损失值
line_avg_critic, = ax2.plot(data['Episode'], data['Moving Average Critic Loss'], color='darkorange', linewidth=2, label='Moving Average Critic Loss')
line_avg_actor, = ax2.plot(data['Episode'], data['Moving Average Actor Loss'], color='darkgreen', linewidth=2, label='Moving Average Actor Loss')

# 损失标准差范围（添加label参数以显示在图例中）
fill_critic = ax2.fill_between(
    data['Episode'],
    data['Moving Average Critic Loss'] - data['Critic Loss Std Dev'],
    data['Moving Average Critic Loss'] + data['Critic Loss Std Dev'],
    color='orange',
    alpha=0.15,
    label='Critic Loss ± Std Dev'
)
fill_actor = ax2.fill_between(
    data['Episode'],
    data['Moving Average Actor Loss'] - data['Actor Loss Std Dev'],
    data['Moving Average Actor Loss'] + data['Actor Loss Std Dev'],
    color='green',
    alpha=0.15,
    label='Actor Loss ± Std Dev'
)

ax2.set_xlabel('Episodes')
ax2.set_ylabel('Loss')
ax2.set_title('Loss of the path recovery agent without Conv')

# 自定义损失图的图例，确保所有6个项目都显示
ax2.legend(handles=[line_critic, line_actor, line_avg_critic, line_avg_actor, fill_critic, fill_actor],
           labels=['Critic Loss', 'Actor Loss', 'Moving Average Critic Loss', 'Moving Average Actor Loss',
                  'Critic Loss ± Std Dev', 'Actor Loss ± Std Dev'],
           loc='upper right')

plt.tight_layout()
plt.show()