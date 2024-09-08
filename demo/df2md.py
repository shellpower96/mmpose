from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# 加载CSV文件并处理
front_df = pd.read_csv('report/正面数据')
left_side_df = pd.read_csv('report/左侧面数据')
right_side_df = pd.read_csv('report/右侧面数据')
back_df = pd.read_csv('report/背面数据')

# 添加标识列
front_df['方向来源'] = '正面'
left_side_df['方向来源'] = '左侧面'
right_side_df['方向来源'] = '右侧面'
back_df['方向来源'] = '背面'

# 合并所有数据
combined_df = pd.concat([front_df, left_side_df, right_side_df, back_df])

# 重置索引
combined_df.reset_index(drop=True, inplace=True)

# 筛选掉指定的评估选项
filter_out_items = ['左肱骨弯曲', '右肱骨弯曲', '肱骨前移']
filtered_df = combined_df[~combined_df['id'].isin(filter_out_items)]

# 按部位(class)和评估内容(id)分组，并选择偏移角度最大的行
grouped_df = filtered_df.loc[filtered_df.groupby(['class', 'id'])['degree'].idxmax()]

# 计算 "正常范围"
grouped_df['正常范围'] = '0' + '~' + (grouped_df['range'] + grouped_df['interval']).astype(str)

# 修改 "左腿Q角" 和 "右腿Q角" 的正常范围
grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']), '正常范围'] = '10~20'

# 参数控制是否显示 "方向来源" 列
show_direction = False

# 选择需要展示的列
columns_to_display = ['class', 'id', 'degree', 'oren', 'level', '正常范围']
if show_direction:
    columns_to_display.insert(-1, '方向来源')  # 在 "正常范围" 前插入 "方向来源"

df = grouped_df[columns_to_display]
df['degree'] = df['degree'].round(1)  # 保留一位小数

# 根据是否显示 "方向来源" 列调整列宽
if show_direction:
    col_widths = [100, 150, 100, 80, 80, 100, 100]  # 显示方向来源时的列宽
else:
    col_widths = [100, 150, 100, 80, 80, 100]  # 不显示方向来源时的列宽
row_height = 40  # 每行的高度
line_width = 2  # 表格线条的宽度
# 创建空白画布
canvas_width = sum(col_widths) + 100
canvas_height = 1500
background_color = (255, 255, 255)  # 白色背景
canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
draw = ImageDraw.Draw(canvas)

# 设置字体
font_title = ImageFont.truetype("demo/heiti.ttf", size=30)  # 标题字体
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=20)  # 表格字体

# 表格起始位置和标题
table_x = 50
table_y = 100
table_title = "体态数据概览"
draw.text((table_x, table_y - 50), table_title, fill="black", font=font_title)

# 文本对齐方式 ('center', 'left', 'right')
text_alignment = 'center'

def draw_text(draw, position, text, font, col_width, row_height, alignment='center'):
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    if alignment == 'center':
        x = position[0] + (col_width - text_width) // 2
    elif alignment == 'left':
        x = position[0] + 5  # 留一些边距
    elif alignment == 'right':
        x = position[0] + col_width - text_width - 5  # 留一些边距
    
    y = position[1] + (row_height - text_height) // 2
    draw.text((x, y), text, fill="black", font=font)
# 绘制表头上方横线
draw.line([(table_x, table_y), (table_x + sum(col_widths), table_y)], fill="black", width=2)
# 绘制表头
headers = ["部位", "评估内容", "偏移角度", "方向", "评分", "正常范围"]
if show_direction:
    headers.insert(-2, "方向来源")  # 在 "正常范围" 前插入 "方向来源"

for i, header in enumerate(headers):
    cell_x = table_x + sum(col_widths[:i])
    draw_text(draw, (cell_x, table_y), header, font_text, col_widths[i], row_height, text_alignment)

# 绘制表头下方的横线
draw.line([(table_x, table_y + row_height), (table_x + sum(col_widths), table_y + row_height)], fill="black", width=2)

# 合并同类部位的单元格
row_y = table_y + row_height
i = 0
while i < len(df):
    start = i
    end = start
    # 找出需要合并的行数
    while end < len(df) - 1 and df.iloc[end]['class'] == df.iloc[end + 1]['class']:
        end += 1
    
    # 合并单元格并居中显示文本
    merge_height = (end - start + 1) * row_height
    cell_x = table_x
    # 只绘制左侧、底部和顶部线条
    draw.line([(cell_x, row_y), (cell_x, row_y + merge_height)], fill="black", width=2)
    draw.line([(cell_x, row_y), (cell_x + col_widths[0], row_y)], fill="black", width=2)
    draw.line([(cell_x, row_y + merge_height), (cell_x + col_widths[0], row_y + merge_height)], fill="black", width=2)

    # 绘制合并单元格的文本
    draw_text(draw, (cell_x, row_y), df.iloc[start]['class'], font_text, col_widths[0], merge_height, text_alignment)
    
    # 绘制其它列
    for j in range(start, end + 1):
        current_row_y = row_y + (j - start) * row_height
        for k, (item, col_width) in enumerate(zip(df.iloc[j, 1:], col_widths[1:])):
            cell_x = table_x + sum(col_widths[:k + 1])
            draw_text(draw, (cell_x, current_row_y), str(item), font_text, col_width, row_height, text_alignment)
            
            # 绘制单元格框线（只绘制右边和下边，避免重复绘制线条）
            if j == start:  # 只在第一次绘制顶部线条，避免重复
                draw.line([(cell_x, current_row_y), (cell_x + col_width, current_row_y)], fill="black", width=2)
            draw.line([(cell_x + col_width, current_row_y), (cell_x + col_width, current_row_y + row_height)], fill="black", width=2)
            draw.line([(cell_x, current_row_y + row_height), (cell_x + col_width, current_row_y + row_height)], fill="black", width=2)
    
    row_y += merge_height
    i = end + 1

# 绘制列线，特别是第一列的右边线
for k in range(len(headers) + 1):
    cell_x = table_x + sum(col_widths[:k])
    draw.line([(cell_x, table_y), (cell_x, row_y)], fill="black", width=2)

# 确保第一列右边的线只绘制一次，不重叠
draw.line([(table_x + col_widths[0], table_y), (table_x + col_widths[0], row_y)], fill="black", width=2)

# 保存最终图像
canvas.save("demo/reports/posture_evaluation_report_direction_toggle.png")
