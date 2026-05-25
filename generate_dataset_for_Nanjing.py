import pandas as pd
import os

DATA_DIR = "data/Xuancheng_GaochunLishui"
dates = ['20250914', '20250920'] # '20230917', '20230923',
DISTANCE_THRESHOLD = 18

output_rows = []

for date in dates:
    filepath = os.path.join(DATA_DIR, f"dataset_{date}_xuancheng_to_gaochun_lishui_with_grid.csv")
    df = pd.read_csv(filepath)
    df['ID'] = date + '_' + df['uid'].astype(str)

    for uid, group in df.groupby('ID'):
        group = group.sort_values('idx')
        rows = group.to_dict('records')

        anchor = rows[0]
        accum_time = 0.0
        accum_dist = 0.0

        for i in range(1, len(rows)):
            cur = rows[i]
            accum_time += cur['time_value']
            accum_dist += cur['dist_value']
            manhattan = abs(cur['loc_x'] - anchor['loc_x']) + abs(cur['loc_y'] - anchor['loc_y'])

            if manhattan >= DISTANCE_THRESHOLD:
                output_rows.append({
                    'ID': uid,
                    'stime_o': pd.to_datetime(anchor['stime'], unit='s').strftime('%H:%M:%S'),
                    'stime_d': pd.to_datetime(cur['stime'], unit='s').strftime('%H:%M:%S'),
                    'lat_o': anchor['lat'],
                    'lon_o': anchor['lon'],
                    'lat_d': cur['lat'],
                    'lon_d': cur['lon'],
                    'locx_o': anchor['loc_x'],
                    'locy_o': anchor['loc_y'],
                    'locx_d': cur['loc_x'],
                    'locy_d': cur['loc_y'],
                    'time': accum_time,
                    'distance': accum_dist,
                    'mode': 'GSD',
                })
                anchor = cur
                accum_time = 0.0
                accum_dist = 0.0

output_df = pd.DataFrame(output_rows)

if not output_df.empty:
    order = ['ID', 'stime_o', 'stime_d', 'lat_o', 'lon_o', 'lat_d', 'lon_d', 'locx_o', 'locy_o', 'locx_d', 'locy_d', 'mode', 'time', 'distance']
    output_df = output_df[order]

output_df.to_csv("data\\Xuancheng_to_GaochunLishui.csv", index=False)
print(f"Output: {len(output_df)} rows saved to Xuancheng_to_GaochunLishui.csv")
