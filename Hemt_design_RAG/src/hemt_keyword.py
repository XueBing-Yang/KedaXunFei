import jieba

hemt_keywords = [
    # ========== 材料关键词 ==========
    # 衬底层材料
    "硅", "Si", "碳化硅", "SiC", "蓝宝石", "Sapphire", "Al₂O₃", 
    "氮化铝", "AlN", "金刚石", "Diamond", "砷化镓", "GaAs", 
    "磷化铟", "InP", "氮化镓", "GaN", "氧化锌", "ZnO", "其它材料",
    "AlGaN/GaN", "AlGaN/GaN", "铝镓氮", 
    "InAlN/GaN", "InAlN/GaN",
    "AlN/GaN", "AlN/GaN"
    
    # 层级结构
    "衬底层",
    "衬底同质外延层", 
    "成核层",
    "缓冲层",
    "背势垒层",
    "沟道层",
    "插入层",
    "势垒层",
    "栅介质层",
    "帽层",
    "钝化层",
    "薄buffer",
    "厚buffer",
    "异质结"

    # 成核层材料
    "AlN", "SiN", "ScN", "GaN", "其它成核材料", "无成核层",
    
    # 缓冲层材料
    "AlGaN", "AlₓGa₁₋ₓN", "InAlN", "InₓAl₁₋ₓN", "AlPN", "AlₓP₁₋ₓN", "无缓冲层",
    
    # 背势垒层材料
    "Fe掺杂GaN", "Fe-doped GaN", "AlGaN", "AlN", "其它背势垒材料", "无背势垒",
    
    # 沟道层材料
    "GaN", "InGaN", "GaAs", "其它沟道材料",
    
    # 插入层材料
    "AlN", "GaN", "无插入层",
    
    # 势垒层材料
    "AlGaN", "AlₓGa₁₋ₓN", "InAlN", "InₓAl₁₋ₓN", "AlPN", "AlₓP₁₋ₓN", "其它势垒材料",
    
    # 栅介质层材料
    "Al₂O₃", "SiO₂", "Si₃N₄", "HfO₂", "其它栅介质材料",
    
    # 帽层材料
    "GaN", "AlN", "其它帽层材料", "无帽层",
    
    # 钝化层材料
    "Si₃N₄", "Al₂O₃", "AlN", "无钝化层",
    
    # ========== 工艺参数关键词 ==========
    # 生长方法
    "提拉法", "Czochralski", "泡生法", "Kyropoulos", "物理气相输运法", "PVT",
    "微波等离子体化学气相沉积", "MPCVD", "水热法", "Hydrothermal", "氨热法", "Ammonothermal",
    "高压高温法", "HPHT", "氢化物气相外延", "HVPE", "金属有机化学气相沉积", "MOCVD",
    "分子束外延", "MBE", "脉冲激光沉积", "PLD", "磁控溅射", "Magnetron Sputtering",
    
    # 工艺参数
    "温度", "Temperature", "气压", "Pressure", "气体流量", "Gas Flow Rate",
    "气体组分比例", "Gas Composition Ratio", "生长速率", "Growth Rate",
    "冷却速率", "Cooling Rate", "V/III比", "V/III Ratio", "反应室压力", "Chamber Pressure",
    "前驱体流量", "Precursor Flow Rate", "退火温度", "Annealing Temperature",
    "退火时间", "Annealing Time", "退火气氛", "Annealing Atmosphere",
    "超声功率", "Ultrasonic Power", "清洗时间", "Cleaning Time",
    "溶液浓度", "Solution Concentration",
    
    # 清洗方法
    "去离子水超声清洗", "DI Water Ultrasonic Cleaning", "有机溶剂清洗", "Organic Solvent Cleaning",
    "酸洗", "Acid Etching", "食人鱼溶液", "Piranha Solution", "磷酸/盐酸混合液", "H₃PO₄/HCl Mixture",
    "等离子体清洗", "Plasma Cleaning",
    
    # 结构参数
    "厚度", "Thickness", "尺寸", "Size", "晶向", "Crystal Orientation", "(100)晶面", "(111)晶面", "(110)晶面",
    "斜切角", "Off-cut Angle", "掺杂浓度", "Doping Concentration", "铝组分", "Al Composition",
    
    # ========== 性能参数关键词 ==========
    # 电学特性
    "载流子密度", "Carrier Density", "方阻", "Sheet Resistance", "迁移率", "Mobility",
    "面电荷密度", "Sheet Charge Density", "导通电阻", "On-resistance", "泄露电流", "Leakage Current",
    "击穿电压", "Breakdown Voltage", "击穿场强", "Breakdown Field", "跨导", "Transconductance",
    "二维电子气浓度", "2DEG Concentration", "阈值电压", "Threshold Voltage", "饱和电流", "Saturation Current",
    
    # 频率特性
    "截止频率", "Cut-off Frequency", "fₜ", "最大振荡频率", "Maximum Oscillation Frequency", "fₘₐₓ",
    "功率增益", "Power Gain",
    
    # 热学特性
    "热阻", "Thermal Resistance", "结温", "Junction Temperature", "热导率", "Thermal Conductivity",
    
    # 可靠性参数
    "寿命测试", "Lifetime Test", "高温工作稳定性", "High Temperature Stability", "栅退化", "Gate Degradation",
    
    # ========== 特殊工艺关键词 ==========
    "刻蚀工艺", "Etching", "干法刻蚀", "Dry Etching", "湿法刻蚀", "Wet Etching",
    "反应离子刻蚀", "RIE", "感应耦合等离子体刻蚀", "ICP", "金属化工艺", "Metallization",
    "欧姆接触", "Ohmic Contact", "肖特基接触", "Schottky Contact", "金属堆叠", "Metal Stack",
    "光刻工艺", "Lithography", "紫外光刻", "UV Lithography", "电子束光刻", "E-beam Lithography",
    "纳米压印", "Nanoimprint",
    
    # ========== 缺陷与表征 ==========
    "位错", "Dislocation", "点缺陷", "Point Defect", "堆垛层错", "Stacking Fault",
    "X射线衍射", "XRD", "扫描电子显微镜", "SEM", "透射电子显微镜", "TEM",
    "原子力显微镜", "AFM", "霍尔测试", "Hall Measurement"
]

if __name__ == "__main__":

    for keyword in hemt_keywords:
        jieba.add_word(keyword)
        jieba.add_word(keyword, freq=10000, tag='n')



    text = "氮化镓HEMT器件使用AlGaN/AlN缓冲层"
    words = jieba.cut(text, cut_all=False)
    print("分词结果:", " | ".join(words))