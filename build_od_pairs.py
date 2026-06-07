import os
import pandas as pd


def build_od_pairs(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_od{ext}"

    rows = []
    for uid, group in df.groupby('uid', sort=False):
        group = group.sort_values('idx').reset_index(drop=True)
        for i in range(len(group) - 1):
            cur = group.iloc[i]
            nxt = group.iloc[i + 1]
            rows.append({
                'ID': str(uid),
                'locxo': int(cur['hex_x']),
                'locyo': int(cur['hex_y']),
                'loczo': int(cur['hex_z']),
                'locxd': int(nxt['hex_x']),
                'locyd': int(nxt['hex_y']),
                'loczd': int(nxt['hex_z']),
                'distance_m': float(nxt['dist_value']),
                'time': int(nxt['time_value']),
                'velocity': float(nxt['velocity']),
            })

    od = pd.DataFrame(rows)
    od.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"输入: {len(df)} 个打点, {df['uid'].nunique()} 个轨迹")
    print(f"输出: {len(od)} 个 OD 对 -> {output_csv}")
    return output_csv


if __name__ == '__main__':
    build_od_pairs(
        'data/dataset_20230917_nanjing_to_gaochun_lishui_with_hex_downsampled.csv'
    )
