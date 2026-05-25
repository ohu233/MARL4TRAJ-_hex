import pandas as pd
import pyproj


def wgs84_to_beijing1954(lon, lat):
    """将WGS84经纬度坐标转换为Beijing 1954 3 Degree GK CM 111E投影坐标"""
    # 根据提供的PRJ配置定义坐标系
    wgs84_prj = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'
    beijing1954_prj = 'PROJCS["Beijing_1954_3_Degree_GK_CM_111E",GEOGCS["GCS_Beijing_1954",DATUM["D_Beijing_1954",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",111.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0],AUTHORITY["EPSG",2434]]'

    wgs84 = pyproj.CRS.from_wkt(wgs84_prj)
    beijing1954 = pyproj.CRS.from_wkt(beijing1954_prj)

    transformer = pyproj.Transformer.from_crs(
        wgs84, beijing1954,
        always_xy=True
    )

    x, y = transformer.transform(lon, lat)
    return x, y


def assign_grid_cell(x, y, x_min, x_max, y_min, y_max, grid_size=1000):
    """
    确定坐标点所属的网格行列
    行和列从0开始计数
    """
    # 确保坐标在有效范围内
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return None, None

    # 计算列号(loc_x)和行号(loc_y)
    loc_x = int((x - x_min) // grid_size)
    loc_y = int((y - y_min) // grid_size)

    return loc_x, loc_y


def process_excel(input_file, output_file):
    """处理Excel文件：转换坐标并划分网格"""
    # 定义网格范围
    # 三省一市
    # x_min, x_max =853241.740909714,1678241.740909714
    # y_min, y_max =3012170.012633881,3956170.012633881
    #
    # 江苏省
    # x_min, x_max =988144.781509014,1552144.781509014
    # y_min, y_max =3417504.294282121,3946504.294282121

    x_min, x_max =988144.781509014,1552144.781509014
    y_min, y_max =3417504.294282121,3946504.294282121
    grid_size = 1000  # 1000米×1000米的网格

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 检查必要的列是否存在
    required_columns = ['lat', 'lon']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV文件必须包含 'lat' 和 'lon' 列")

    # 初始化结果列
    df['x'] = None
    df['y'] = None
    df['loc_x'] = None
    df['loc_y'] = None

    # 批量处理每一行
    for index, row in df.iterrows():
        try:
            # 转换坐标
            lon = row['lon']
            lat = row['lat']
            x, y = wgs84_to_beijing1954(lon, lat)

            # 计算网格位置
            loc_x, loc_y = assign_grid_cell(x, y, x_min, x_max, y_min, y_max, grid_size)

            # 保存结果
            df.at[index, 'x'] = round(x, 3)
            df.at[index, 'y'] = round(y, 3)
            df.at[index, 'loc_x'] = loc_x
            df.at[index, 'loc_y'] = loc_y

        except Exception as e:
            print(f"处理第{index + 1}行时出错: {str(e)}")

    # 保存结果到新的CSV文件
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")
    print(f"网格范围：")
    print(f"X方向: {x_min}m - {x_max}m，共 {(x_max - x_min) // grid_size} 列")
    print(f"Y方向: {y_min}m - {y_max}m，共 {(y_max - y_min) // grid_size} 行")


if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = "dataset_multicity_20230917_processed.csv"  # 替换为您的输入文件路径
    output_csv = "坐标转换后-" + input_csv  # 输出文件路径

    # 执行处理
    process_excel(input_csv, output_csv)
