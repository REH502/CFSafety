import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joypy


def correct_percent(LLM_name):
    base = f'{LLM_name}/distribution'
    path = os.listdir(base)
    correct = []
    # 创建一个空的DataFrame
    combined_data = pd.DataFrame()

    # 加载每个文件
    for file in path:
        if file.endswith('.csv'):
            # 读取指定列，这里假设列名是'target_col'
            df = pd.read_csv(os.path.join(base, file), usecols=['score'])
            # 将列名更改为文件名，用作类别标签
            df.rename(columns={f'{file}': file[:-4]}, inplace=True)
            # 将数据合并到一个DataFrame
            combined_data = pd.concat([combined_data, df], axis=1)
    combined_data.columns = ['Criminal Unethical', 'Insults SensitiveTopics', 'LLMRewriting', 'MinorityLanguage',
                             'Misleading', 'PrivacyLeaks',
                             'PromptLeakage', 'ReverseExposure', 'ScenarioEmbedding', 'SocialBias']
    print(combined_data.mean())

    # 绘制Joy Plot
    fig, axes = joypy.joyplot(combined_data, x_range=(-1, 6))
    plt.suptitle(f"Distribution of {LLM_name}'s Safety Score", fontsize=15, fontname='Arial', x=0.5, y=0.93)
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=10)
    plt.savefig(f'{LLM_name}.png')
    plt.show()
    for type_file in path:
        data = pd.read_csv(os.path.join(base, type_file))
        type_correct = data['yn']
        correct.extend(type_correct)

    percent = np.array(correct).mean()
    print(percent)


