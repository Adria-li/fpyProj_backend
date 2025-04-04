from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import nolds

app = Flask(__name__)
CORS(app, origins=["https://fpy-proj-frontend.vercel.app"])

@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求体为空"}), 400

        # 获取所有键值对，并确保值是列表
        # 获取所有的 key
        keys = list(data.keys())

        print("收到的 keys:", keys)  # 打印调试信息
        
        # 确保至少有两个键
        if len(keys) < 2:
            return jsonify({"error": "至少需要两个数组"}), 400

        # 校验所有数组的长度
        for key in keys:
            if len(np.array(data[key])) < 100:
                return jsonify({"error": f"'{key}' 长度必须至少为 100"}), 400
        
        res1 = np.array(data[keys[0]])
        res2 = np.array(data[keys[1]])
            
        # 滑动窗口的长度和步长
        window_size, step_size = 100, 6
        all_lyapunov_results = {"window_size": window_size, "step_size": step_size, "results": {}}
        raw_lyapunov_results = {}

        def calculate_output(series, index):
            # 确保 index 有效
            keys = list(data.keys())
            if index >= len(keys):
                return jsonify({"error": f"索引 {index} 超出可用 keys 范围"}), 400

            key = keys[index]
            lyapunov_results = []

            # 滑动窗口计算 Lyapunov 指数
            for start_idx in range(0, len(series) - window_size + 1, step_size):
                # print("start_idx:", start_idx)
                
                window_series = series[start_idx:start_idx + window_size]
                lyap_exponents = calculate_lyapunov(window_series)
                lyapunov_results.append(lyap_exponents)

            raw_lyapunov_results[key] = {
                "occupancy_time_series": data[key],  # 存储原始时间序列
                "lyapunov_exponents": lyapunov_results
            }

        calculate_output(res1, 0)
        calculate_output(res2, 1)

        # 进行标准归一化处理
        standardized_results = min_max_normalize_lyapunov_fixed_range(raw_lyapunov_results)

        # 存储完整的结果
        all_lyapunov_results["results"] = standardized_results
        # print("all_lyapunov_results type:", type(jsonify(all_lyapunov_results)))
        return jsonify(all_lyapunov_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_lyapunov(occupancy_time_series, emb_dim=6, matrix_dim=3):
    """计算时间序列的最大 Lyapunov 指数"""
    # 转换为 NumPy 数组并清理数据
    occupancy_time_series = np.array(occupancy_time_series, dtype=float)
    occupancy_time_series = occupancy_time_series[~np.isnan(occupancy_time_series)]  # 移除 NaN
    occupancy_time_series = occupancy_time_series[~np.isinf(occupancy_time_series)]  # 移除 Inf

    # 确保 emb_dim 和 matrix_dim 满足条件
    while (emb_dim - 1) % (matrix_dim - 1) != 0:
        emb_dim += 1

    # 计算 Lyapunov 指数谱
    lyap_exponents = nolds.lyap_e(occupancy_time_series, emb_dim=emb_dim, matrix_dim=matrix_dim)
    largest_lyap = lyap_exponents[0]  # 最大 Lyapunov 指数
    return largest_lyap

# def min_max_normalize_lyapunov(all_lyapunov_results):
#     """对所有 detector_id 的最大 Lyapunov 指数进行 Min-Max 归一化（所有 ID 共享同一尺度）"""
#     all_values = []
    
#     # **收集所有 detector_id 的 Lyapunov 指数**
#     for det in all_lyapunov_results.values():
#         all_values.extend(det["lyapunov_exponents"])  # 统一存入列表

#     if not all_values:
#         return all_lyapunov_results  # 如果为空，则不做归一化
    
#     print("all_values:", all_values)
    
#     # filtered_exponents = all_values[np.isfinite(all_values)].tolist()
#     filtered_exponents = [x for x in all_values if np.isfinite(x)]
#     print("filtered_exponents:", filtered_exponents)

#     # **计算全局 min 和 max**
#     min_value = min(filtered_exponents)
#     max_value = max(filtered_exponents)

#     # **打印 min 和 max 值**
#     print(f"Global Min Lyapunov: {min_value}")
#     print(f"Global Max Lyapunov: {max_value}")

#     if max_value == min_value:  # 避免除零错误
#         for det_id, det in all_lyapunov_results.items():
#             det["normalized_lyapunov_exponents"] = [0.5 for _ in det["lyapunov_exponents"]]  # 若所有值相同，则设为 0.5
#         return all_lyapunov_results

#     # **进行全局归一化**
#     for det_id, det in all_lyapunov_results.items():
#         det["normalized_lyapunov_exponents"] = [
#             (value - min_value) / (max_value - min_value) for value in det["lyapunov_exponents"]
#         ]

#     return all_lyapunov_results

def min_max_normalize_lyapunov_fixed_range(all_lyapunov_results, min_value=-1.0, max_value=1.0):
    for det_id, det in all_lyapunov_results.items():
        det["normalized_lyapunov_exponents"] = [
            max(0.0, min(1.0, (x - min_value) / (max_value - min_value))) if np.isfinite(x) else 0.5
            for x in det["lyapunov_exponents"]
        ]
    return all_lyapunov_results


if __name__ == '__main__':
    app.run(debug=True, port=5000)