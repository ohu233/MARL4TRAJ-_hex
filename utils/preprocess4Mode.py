import pandas as pd
from collections import defaultdict

INPUT_PATH = r"data\data_lower_train.csv"
OUTPUT_PATH = r"data\data_lower_train_ordered.csv"


def order_one_group(g: pd.DataFrame) -> pd.DataFrame:
    """
    组内按规则重排：
    当前行的 (locx_d, locy_d) 去匹配下一行的 (locx_o, locy_o)。
    若组内存在多条链/断链，按链依次拼接，并保证所有行都被使用一次。
    """
    g = g.copy().reset_index(drop=True)

    # 每一行是一条边：origin -> destination
    origins = list(zip(g["locx_o"], g["locy_o"]))
    dests = list(zip(g["locx_d"], g["locy_d"]))

    # origin 点 -> 该点出发的行号列表
    origin_to_rows = defaultdict(list)
    for i, o in enumerate(origins):
        origin_to_rows[o].append(i)

    dest_set = set(dests)
    unvisited = set(range(len(g)))
    ordered_idx = []

    def pop_next_by_origin(origin_point):
        """从未访问行里找一条 origin=origin_point 的行。"""
        cand = origin_to_rows.get(origin_point, [])
        while cand and cand[0] not in unvisited:
            cand.pop(0)
        if not cand:
            return None
        idx = cand.pop(0)
        return idx

    while unvisited:
        # 优先找链头：其 origin 不在任何 destination 里
        start = None
        for i in list(unvisited):
            if origins[i] not in dest_set:
                start = i
                break

        # 如果全是环或都可回指，随便取一个未访问点作为起点
        if start is None:
            start = next(iter(unvisited))

        # 从 start 往后串
        cur = start
        while cur is not None and cur in unvisited:
            ordered_idx.append(cur)
            unvisited.remove(cur)

            next_origin = dests[cur]
            nxt = pop_next_by_origin(next_origin)
            if nxt is not None and nxt in unvisited:
                cur = nxt
            else:
                break

    return g.iloc[ordered_idx].reset_index(drop=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    # 保持组顺序（sort=False）
    out_parts = []
    for gid, g in df.groupby("ID", sort=False):
        ordered_g = order_one_group(g)
        out_parts.append(ordered_g)

    out_df = pd.concat(out_parts, ignore_index=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"done: {OUTPUT_PATH}, rows={len(out_df)}")


if __name__ == "__main__":
    main()
