from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
# 创建空白画布
canvas_width = 1654  # 对应 210mm
canvas_height = 2339  # 对应 297mm
background_color = (255, 255, 255)  # 白色背景
canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

# 初始化绘图
draw = ImageDraw.Draw(canvas)

# 加载字体
font_title = ImageFont.truetype("demo/heiti.ttf", size=60)  # 标题字体
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=30)   # 文字字体

# 添加 logo
logo = Image.open("demo/resources/new_logo_white.png").convert("RGBA")
logo_width, logo_height = logo.size

# 调整 logo 大小和位置
logo_ratio = 0.1  # 以画布宽度的10%为基准调整 logo 大小
new_logo_width = int(canvas_width * logo_ratio)
new_logo_height = int(logo_height * (new_logo_width / logo_width))
logo = logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)

# 设置 logo 和标题在同一行，计算 logo 的位置
logo_x = int(canvas_width * 0.05)
logo_y = int(canvas_height * 0.02)
canvas.paste(logo, (logo_x, logo_y), mask=logo)

# 添加标题并水平居中
title = "体态评估测量报告"
title_bbox = draw.textbbox((0, 0), title, font=font_title)
title_width = title_bbox[2] - title_bbox[0]
title_x = (canvas_width - title_width) // 2
title_y = logo_y + (new_logo_height - title_bbox[3] + title_bbox[1]) // 2  # 与 logo 垂直居中
draw.text((title_x, title_y), title, fill="black", font=font_title)

# 获取当前日期并右对齐，在标题下一行
current_date = datetime.now().strftime("%Y-%m-%d")
current_date = '测量日期：' + current_date
date_bbox = draw.textbbox((0, 0), current_date, font=font_text)
date_width = date_bbox[2] - date_bbox[0]
date_x = canvas_width - date_width - int(canvas_width * 0.05)  # 距离右边缘5%的边距
date_y = title_y + title_bbox[3] - title_bbox[1] + 20  # 在标题下方，间距20px
draw.text((date_x, date_y), current_date, fill="black", font=font_text)

# 示例数据
name = "小明"
age = 22
height = 180
weight = 70
bmi = 21.6
sex = '男'

# 设置数据行的起始位置
data_y = date_y + 50
data_x = int(canvas_width * 0.05)  # 距离左边缘5%的边距
draw.line([(data_x, data_y - 5), (canvas_width - data_x, data_y - 5)], fill="black", width=3)

filter_line_font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=30)   # 文字字体

# 计算每个数据块的宽度
data_items = [f"姓名: {name}", f"性别：{sex}", f"年龄: {age}", f"身高: {height}cm", f"体重: {weight}kg", f"BMI: {bmi}"]
total_width = canvas_width - 2 * data_x  # 总宽度
spacing = (total_width - sum(draw.textbbox((0, 0), item, font=filter_line_font_text)[2] - draw.textbbox((0, 0), item, font=filter_line_font_text)[0] for item in data_items)) // (len(data_items) - 1)  # 间隔计算

# 均匀分布数据
current_x = data_x
for item in data_items:
    draw.text((current_x, data_y), item, fill="black", font=filter_line_font_text)
    item_bbox = draw.textbbox((0, 0), item, font=filter_line_font_text)
    item_width = item_bbox[2] - item_bbox[0]
    current_x += item_width + spacing

# 获取数据行高度并画数据行下方的分隔线
data_line_height = draw.textbbox((0, 0), data_items[0], font=filter_line_font_text)[3] - draw.textbbox((0, 0), data_items[0], font=filter_line_font_text)[1]
data_y_end = data_y + data_line_height + 5
draw.line([(data_x, data_y_end), (canvas_width - data_x, data_y_end)], fill="black", width=3)

# 假设这些是照片的路径
photo_paths = [
    "demo/mybody/res/demo/mybody/正面.jpg",  # 正面
    "demo/mybody/res/demo/mybody/左侧面.jpg",   # 左侧面
    "demo/mybody/res/demo/mybody/右侧面.jpg",  # 右侧面
    "demo/mybody/res/demo/mybody/背面.jpg"    # 背面
]

# 照片对应的名称
photo_names = ["正面", "左侧面", "右侧面", "背面"]
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=20)

# 最大宽度和高度
max_width = 300
max_height = 500

# 加载并按照比例缩放照片
photos = []
for photo_path in photo_paths:
    photo = Image.open(photo_path).convert("RGB")
    ratio = min(max_width / photo.width, max_height / photo.height)
    new_size = (int(photo.width * ratio), int(photo.height * ratio))
    photo = photo.resize(new_size, Image.Resampling.LANCZOS)
    photos.append(photo)

# 计算照片的总宽度
total_photos_width = sum(photo.width for photo in photos)

# 计算每张照片之间的间隔
spacing = (canvas_width - total_photos_width) // (len(photos) + 1)

# 起始的 Y 坐标
photos_y = data_y_end + 30

# 绘制照片并添加名称
current_x = spacing
for i, photo in enumerate(photos):
    # 计算每张照片的 X 坐标
    photo_x = current_x
    canvas.paste(photo, (photo_x, photos_y))  # 将照片粘贴到画布上
    # 在照片下方添加名称
    name_x = photo_x + (photo.width - draw.textbbox((0, 0), photo_names[i], font=font_text)[2]) // 2
    name_y = photos_y + photo.height + 10
    draw.text((name_x, name_y), photo_names[i], fill="black", font=font_text)
    # 更新下一个照片的 X 坐标
    current_x += photo.width + spacing

middle_x = canvas_width // 2

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
filter_out_items = ['左肱骨弯曲', '右肱骨弯曲', '左肱骨位置', '右肱骨位置']
filtered_df = combined_df[~combined_df['id'].isin(filter_out_items)]

# 按部位(class)和评估内容(id)分组，并选择偏移角度最大的行
grouped_df = filtered_df.loc[filtered_df.groupby(['class', 'id'])['degree'].idxmax()]

# 计算 "正常范围"
grouped_df['正常范围'] = '0' + '~' + (grouped_df['range'] + grouped_df['interval']).astype(str)

# 修改 "左腿Q角" 和 "右腿Q角" 的正常范围
grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']), '正常范围'] = '10~20'

# 按照规定的部位顺序进行排序
order = ['头颈部', '肩部', '躯干', '骨盆', '腿部', '脚部']
grouped_df['order'] = grouped_df['class'].map({name: i for i, name in enumerate(order)})
sorted_df = grouped_df.sort_values('order').drop('order', axis=1)

# 参数控制是否显示 "方向来源" 列
show_direction = False

# 选择需要展示的列
columns_to_display = ['class', 'id', 'degree', 'oren', 'level', '正常范围']
if show_direction:
    columns_to_display.insert(-1, '方向来源')  # 在 "正常范围" 前插入 "方向来源"
df = sorted_df[columns_to_display]
df['degree'] = df['degree'].round(1)  # 保留一位小数

# 根据是否显示 "方向来源" 列调整列宽
if show_direction:
    col_widths = [100, 150, 100, 80, 80, 100, 100]  # 显示方向来源时的列宽
else:
    col_widths = [100, 150, 100, 80, 80, 100]  # 不显示方向来源时的列宽
col_widths = [c_d +30 for c_d in col_widths]
row_height = 50  # 每行的高度
line_width = 2  # 表格线条的宽度

# 设置字体
font_title = ImageFont.truetype("demo/heiti.ttf", size=40)  # 标题字体
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=30)  # 表格字体

# 表格起始位置和标题
table_x = 50
table_y = name_y + 100  # 调整表格的起始位置
# table_title = "体态数据概览"
# draw.text((table_x, table_y - 50), table_title, fill="black", font=font_title)
# 计算体态数据概览标题位置
table_content_width = sum(col_widths)
table_title = "体态数据概览"
table_title_bbox = draw.textbbox((0, 0), table_title, font=font_title)
table_title_x = table_x + (table_content_width - (table_title_bbox[2] - table_title_bbox[0])) // 2
table_title_y = table_y - 50
draw.text((table_title_x, table_title_y), table_title, fill="black", font=font_title)

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
end_point_x =  cell_x
# 绘制列线，特别是第一列的右边线
for k in range(len(headers) + 1):
    cell_x = table_x + sum(col_widths[:k])
    draw.line([(cell_x, table_y), (cell_x, row_y)], fill="black", width=2)

# 确保第一列右边的线只绘制一次，不重叠
draw.line([(table_x + col_widths[0], table_y), (table_x + col_widths[0], row_y)], fill="black", width=2)

# 右半部分放置人体图片
def black_to_transparent(image_path, output_path):
    """
    替换图像中的黑色背景为透明背景
    """
    # 打开图像
    image = Image.open(image_path).convert("RGBA")
    
    # 获取图像数据
    data = image.getdata()
    
    new_data = []
    for item in data:
        # 替换黑色背景为透明
        if item[:3] == (0, 0, 0):
            new_data.append((0, 0, 0, 0))  # 透明
        else:
            new_data.append(item)
    
    # 更新图像数据
    image.putdata(new_data)
    
    # 保存新的图像
    image.save(output_path, "PNG")


# Function to crop the image based on non-black pixels
def process_image_and_keypoints(skeleton_img, kpts, target_width=300):
    # Step 1: Crop the image to remove black borders
    # def crop_to_content(kpts):
    #     # 找到关键点的最小外接矩形
    #     x_min, y_min = np.min(kpts, axis=0)
    #     x_max, y_max = np.max(kpts, axis=0)
    #     # non_black_mask = np.any(image_array[:, :, :3] != 0, axis=2)
    #     # coords = np.argwhere(non_black_mask)
    #     # y0, x0 = coords.min(axis=0)
    #     # y1, x1 = coords.max(axis=0) + 1  # Add 1 to include the last pixel
    #     return x_min, y_min, x_max, y_max
    x_min, y_min = np.min(kpts, axis=0)
    x_max, y_max = np.max(kpts, axis=0)
    # 裁剪图像，去除黑色背景
    cropped_img = skeleton_img[int(y_min):int(y_max), int(x_min):int(x_max)]

    # 计算宽度的缩放比例，预设宽度为 300
    preset_width = 300
    scale = preset_width / cropped_img.shape[1]

    # 缩放图像到预设宽度
    new_height = int(cropped_img.shape[0] * scale)
    resized_img = cv2.resize(cropped_img, (preset_width, new_height))

    # 调整关键点坐标，根据裁剪和缩放比例
    scaled_kpts = (kpts - [x_min, y_min]) * scale



    return resized_img, scaled_kpts


skeleton_img_path = "demo/resources/skeleton_org.png"
body_image_path = "demo/resources/skeleton.png"
skeleton_img = cv2.imread(skeleton_img_path)
full_kpts = np.load('demo/resources/full_kpts.npy')
target_width = 300
skeleton_img, kpts = process_image_and_keypoints(skeleton_img, full_kpts, target_width)

cv2.imwrite(body_image_path, skeleton_img)
black_to_transparent(body_image_path, body_image_path)
if sex == "男":
    body_image_path = "demo/resources/skeleton.png"
else:
    body_image_path = "demo/resources/skeleton.png"





print("体态问题展示")
grouped_df = combined_df.loc[combined_df.groupby(['class', 'id'])['degree'].idxmax()]
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'degree'].idxmin()], inplace=True)
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左肱骨位置', '右肱骨位置']),'degree'].idxmin()], inplace=True)
# grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左膝超伸', '右膝超伸']),'degree'].idxmin()], inplace=True)
# grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左重心前移', '右重心前移']),'degree'].idxmin()], inplace=True)
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['头部倾斜', '头部水平']),'degree'].idxmin()], inplace=True)
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左肱骨弯曲', '右肱骨弯曲']),'degree'].idxmin()], inplace=True)
grouped_df.loc[grouped_df['id'].isin(['左肱骨位置', '右肱骨位置']),'id'] = '圆肩'
grouped_df.loc[grouped_df['id'].isin(['左肱骨弯曲', '右肱骨弯曲']),'id'] = '肱骨前移'
grouped_df.loc[grouped_df['id']=='头部倾斜','id'] = '头倾斜'
grouped_df.loc[grouped_df['id']=='颈椎倾斜','id'] = '头前引'
grouped_df.loc[grouped_df['id']=='双肩水平','id'] = '高低肩'
oren = grouped_df.loc[grouped_df['id']=='骨盆前/后倾','oren']
grouped_df.loc[grouped_df['id']=='骨盆前/后倾','id'] = '骨盆{}倾'.format(oren.values[0])
oren = grouped_df.loc[grouped_df['id']=='骨盆前/后移','oren']
grouped_df.loc[grouped_df['id']=='骨盆前/后移','id'] = '骨盆{}移'.format(oren.values[0])
grouped_df.loc[grouped_df['id']=='骨盆倾斜','id'] = '骨盆侧倾'
grouped_df = grouped_df[~grouped_df['id'].isin(['膝关节水平', '踝关节水平'])]
oren = grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'oren']
if oren.values[0] == '内':
    grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'id'] = 'X型腿'
else:
    grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'id'] = 'O型腿'
oren = grouped_df.loc[grouped_df['id']=='左足内/外翻','oren']
grouped_df.loc[grouped_df['id']=='左足内/外翻','id'] = '左足{}翻'.format(oren.values[0])
oren = grouped_df.loc[grouped_df['id']=='右足内/外翻','oren']
grouped_df.loc[grouped_df['id']=='右足内/外翻','id'] = '右足{}翻'.format(oren.values[0])
# 将严重程度转换为风险等级
risk_mapping = {
    '正常': '无风险',
    '轻微': '低风险',
    '明显': '中风险',
    '严重': '高风险'
}
grouped_df['risk_level'] = grouped_df['level'].map(risk_mapping)
order = ['头颈部', '肩部', '躯干', '骨盆', '腿部', '脚部']
grouped_df['order'] = grouped_df['class'].map({name: i for i, name in enumerate(order)})
grouped_df = grouped_df.sort_values('order').drop('order', axis=1)

# 获取评估内容和转换后的风险等级
issues = {}
for index, row in grouped_df.iterrows():
    issue = row['id']
    risk_level = row['risk_level']
    issues[f'issue_{index}'] = (risk_level, issue)
# 选择需要展示的列
columns_to_display = ['class', 'id', 'degree', 'oren', 'level', 'risk_level']
df.to_csv('report/full.csv')
df = grouped_df[columns_to_display]
df['degree'] = df['degree'].round(1)  # 保留一位小数

text_x = table_x + sum(col_widths)
text_y = name_y + 50
body_image = Image.open(body_image_path).convert("RGBA")

comprehensive_title = "体态综合概览"
comprehensive_title_bbox = draw.textbbox((0, 0), comprehensive_title, font=font_title)
comprehensive_title_x = text_x + (canvas_width - text_x - (comprehensive_title_bbox[2] - comprehensive_title_bbox[0])) // 2
comprehensive_title_y = text_y
draw.text((comprehensive_title_x, comprehensive_title_y), comprehensive_title, fill="black", font=font_title)

# body_title = "体态综合概览"
# draw.text((text_x, text_y), body_title, fill="black", font=font_title)


font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=25)  # 表格字体
# 调整人体图片大小，保持透明背景
body_image_ratio = 0.5
body_image = body_image.resize((int(body_image.width * body_image_ratio), int(body_image.height * body_image_ratio)), Image.LANCZOS)
body_image_x = text_x + (canvas_width // 2 - body_image.width) // 2
body_image_y = text_y +150 
canvas.paste(body_image, (body_image_x, body_image_y), body_image)

# 画布设置
start_x = body_image_x - body_image.width - 100
start_y = body_image_y - 50
column_spacing = body_image.width * 2 + 200  # 两列之间的间隔
row_spacing = 20  # 每个方框之间的垂直间隔
box_width = 150  # 每个方框的宽度
box_height = 70  # 每个方框的高度

# 计算每列可以放置的方框数量
num_rows = (body_image.height - start_y) // (box_height + row_spacing)
num_columns = 2

# 背景颜色变量
background_color = (173, 216, 230)  # 浅蓝色背景

# 风险等级到颜色的映射
color_mapping = {
    '无风险': (0, 128, 0),      # 绿色
    '低风险': (255, 255, 0),    # 浅黄色
    '中风险': (255, 165, 0),    # 橘黄色
    '高风险': (255, 0, 0)       # 红色
}

x17, y17 = body_image_x + body_image_ratio* kpts[[17],0], body_image_y + body_image_ratio* kpts[[17],1] #head
x18, y18 = body_image_x + body_image_ratio* kpts[[18],0], body_image_y + body_image_ratio* kpts[[18],1] #neck
x19, y19 = body_image_x + body_image_ratio* kpts[[19],0], body_image_y + body_image_ratio* kpts[[19],1] #neck
x5, y5 = body_image_x + body_image_ratio* kpts[[5],0], body_image_y + body_image_ratio* kpts[[5],1] #shoulder_right
x6, y6 = body_image_x + body_image_ratio* kpts[[6],0], body_image_y + body_image_ratio* kpts[[6],1] #shoulder_left
x9, y9 = body_image_x + body_image_ratio* kpts[[9],0], body_image_y + body_image_ratio* kpts[[9],1] #hand_right
x10, y10 = body_image_x + body_image_ratio* kpts[[10],0], body_image_y + body_image_ratio* kpts[[10],1] #hand_left
x11, y11 = body_image_x + body_image_ratio* kpts[[11],0], body_image_y + body_image_ratio* kpts[[11],1] #hip_right
x12, y12 = body_image_x + body_image_ratio* kpts[[12],0], body_image_y + body_image_ratio* kpts[[12],1] #hip_left
x9, y9 = body_image_x + body_image_ratio* kpts[[9],0], body_image_y + body_image_ratio* kpts[[9],1]
x10, y10 = body_image_x + body_image_ratio* kpts[[10],0], body_image_y + body_image_ratio* kpts[[10],1]
x14, y14 = body_image_x + body_image_ratio* kpts[[14],0], body_image_y + body_image_ratio* kpts[[14],1]
x13, y13 = body_image_x + body_image_ratio* kpts[[13],0], body_image_y + body_image_ratio* kpts[[13],1]
x16, y16 = body_image_x + body_image_ratio* kpts[[16],0], body_image_y + body_image_ratio* kpts[[16],1]
x15, y15 = body_image_x + body_image_ratio* kpts[[15],0], body_image_y + body_image_ratio* kpts[[15],1]


# 示例数组，实际应用中请根据你的需求定义合适的坐标
start_points = [
    (x17, y17),
    (x18, y18),
    (x6, y6),
    (x5, y5),
    (x6, y6),
    (x19, y18+(y19-y18)//2),
    (x19, y19),
    (x11, y11),
    (x12, y12),
    (x11, y11),
    (x14, y14),
    (x13, y13),
    (x16, y16),
    (x15, y15),
]

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
# 创建空白画布
canvas_width = 1654  # 对应 210mm
canvas_height = 2339  # 对应 297mm
background_color = (255, 255, 255)  # 白色背景
canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

# 初始化绘图
draw = ImageDraw.Draw(canvas)

# 加载字体
font_title = ImageFont.truetype("demo/heiti.ttf", size=60)  # 标题字体
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=30)   # 文字字体

# 添加 logo
logo = Image.open("demo/resources/new_logo_white.png").convert("RGBA")
logo_width, logo_height = logo.size

# 调整 logo 大小和位置
logo_ratio = 0.1  # 以画布宽度的10%为基准调整 logo 大小
new_logo_width = int(canvas_width * logo_ratio)
new_logo_height = int(logo_height * (new_logo_width / logo_width))
logo = logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)

# 设置 logo 和标题在同一行，计算 logo 的位置
logo_x = int(canvas_width * 0.05)
logo_y = int(canvas_height * 0.02)
canvas.paste(logo, (logo_x, logo_y), mask=logo)

# 添加标题并水平居中
title = "体态评估测量报告"
title_bbox = draw.textbbox((0, 0), title, font=font_title)
title_width = title_bbox[2] - title_bbox[0]
title_x = (canvas_width - title_width) // 2
title_y = logo_y + (new_logo_height - title_bbox[3] + title_bbox[1]) // 2  # 与 logo 垂直居中
draw.text((title_x, title_y), title, fill="black", font=font_title)

# 获取当前日期并右对齐，在标题下一行
current_date = datetime.now().strftime("%Y-%m-%d")
current_date = '测量日期：' + current_date
date_bbox = draw.textbbox((0, 0), current_date, font=font_text)
date_width = date_bbox[2] - date_bbox[0]
date_x = canvas_width - date_width - int(canvas_width * 0.05)  # 距离右边缘5%的边距
date_y = title_y + title_bbox[3] - title_bbox[1] + 20  # 在标题下方，间距20px
draw.text((date_x, date_y), current_date, fill="black", font=font_text)

# 示例数据
name = "小明"
age = 22
height = 180
weight = 70
bmi = 21.6
sex = '男'

# 设置数据行的起始位置
data_y = date_y + 50
data_x = int(canvas_width * 0.05)  # 距离左边缘5%的边距
draw.line([(data_x, data_y - 5), (canvas_width - data_x, data_y - 5)], fill="black", width=3)

filter_line_font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=30)   # 文字字体

# 计算每个数据块的宽度
data_items = [f"姓名: {name}", f"性别：{sex}", f"年龄: {age}", f"身高: {height}cm", f"体重: {weight}kg", f"BMI: {bmi}"]
total_width = canvas_width - 2 * data_x  # 总宽度
spacing = (total_width - sum(draw.textbbox((0, 0), item, font=filter_line_font_text)[2] - draw.textbbox((0, 0), item, font=filter_line_font_text)[0] for item in data_items)) // (len(data_items) - 1)  # 间隔计算

# 均匀分布数据
current_x = data_x
for item in data_items:
    draw.text((current_x, data_y), item, fill="black", font=filter_line_font_text)
    item_bbox = draw.textbbox((0, 0), item, font=filter_line_font_text)
    item_width = item_bbox[2] - item_bbox[0]
    current_x += item_width + spacing

# 获取数据行高度并画数据行下方的分隔线
data_line_height = draw.textbbox((0, 0), data_items[0], font=filter_line_font_text)[3] - draw.textbbox((0, 0), data_items[0], font=filter_line_font_text)[1]
data_y_end = data_y + data_line_height + 5
draw.line([(data_x, data_y_end), (canvas_width - data_x, data_y_end)], fill="black", width=3)

# 假设这些是照片的路径
photo_paths = [
    "demo/results/p1.jpg",  # 正面
    "demo/results/p2.jpg",   # 左侧面
    "demo/results/p2.jpg",  # 右侧面
    "demo/results/p3.jpg"    # 背面
]

# 照片对应的名称
photo_names = ["正面", "左侧面", "右侧面", "背面"]
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=20)

# 最大宽度和高度
max_width = 300
max_height = 500

# 加载并按照比例缩放照片
photos = []
for photo_path in photo_paths:
    photo = Image.open(photo_path).convert("RGB")
    ratio = min(max_width / photo.width, max_height / photo.height)
    new_size = (int(photo.width * ratio), int(photo.height * ratio))
    photo = photo.resize(new_size, Image.Resampling.LANCZOS)
    photos.append(photo)

# 计算照片的总宽度
total_photos_width = sum(photo.width for photo in photos)

# 计算每张照片之间的间隔
spacing = (canvas_width - total_photos_width) // (len(photos) + 1)

# 起始的 Y 坐标
photos_y = data_y_end + 30

# 绘制照片并添加名称
current_x = spacing
for i, photo in enumerate(photos):
    # 计算每张照片的 X 坐标
    photo_x = current_x
    canvas.paste(photo, (photo_x, photos_y))  # 将照片粘贴到画布上
    # 在照片下方添加名称
    name_x = photo_x + (photo.width - draw.textbbox((0, 0), photo_names[i], font=font_text)[2]) // 2
    name_y = photos_y + photo.height + 10
    draw.text((name_x, name_y), photo_names[i], fill="black", font=font_text)
    # 更新下一个照片的 X 坐标
    current_x += photo.width + spacing

middle_x = canvas_width // 2

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
filter_out_items = ['左肱骨弯曲', '右肱骨弯曲', '左肱骨位置', '右肱骨位置']
filtered_df = combined_df[~combined_df['id'].isin(filter_out_items)]

# 按部位(class)和评估内容(id)分组，并选择偏移角度最大的行
grouped_df = filtered_df.loc[filtered_df.groupby(['class', 'id'])['degree'].idxmax()]

# 计算 "正常范围"
grouped_df['正常范围'] = '0' + '~' + (grouped_df['range'] + grouped_df['interval']).astype(str)

# 修改 "左腿Q角" 和 "右腿Q角" 的正常范围
grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']), '正常范围'] = '10~20'

# 按照规定的部位顺序进行排序
order = ['头颈部', '肩部', '躯干', '骨盆', '腿部', '脚部']
grouped_df['order'] = grouped_df['class'].map({name: i for i, name in enumerate(order)})
sorted_df = grouped_df.sort_values('order').drop('order', axis=1)

# 参数控制是否显示 "方向来源" 列
show_direction = False

# 选择需要展示的列
columns_to_display = ['class', 'id', 'degree', 'oren', 'level', '正常范围']
if show_direction:
    columns_to_display.insert(-1, '方向来源')  # 在 "正常范围" 前插入 "方向来源"
df = sorted_df[columns_to_display]
df['degree'] = df['degree'].round(1)  # 保留一位小数

# 根据是否显示 "方向来源" 列调整列宽
if show_direction:
    col_widths = [100, 150, 100, 80, 80, 100, 100]  # 显示方向来源时的列宽
else:
    col_widths = [100, 150, 100, 80, 80, 100]  # 不显示方向来源时的列宽
col_widths = [c_d +30 for c_d in col_widths]
row_height = 50  # 每行的高度
line_width = 2  # 表格线条的宽度

# 设置字体
font_title = ImageFont.truetype("demo/heiti.ttf", size=40)  # 标题字体
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=30)  # 表格字体

# 表格起始位置和标题
table_x = 50
table_y = name_y + 100  # 调整表格的起始位置
# table_title = "体态数据概览"
# draw.text((table_x, table_y - 50), table_title, fill="black", font=font_title)
# 计算体态数据概览标题位置
table_content_width = sum(col_widths)
table_title = "体态数据概览"
table_title_bbox = draw.textbbox((0, 0), table_title, font=font_title)
table_title_x = table_x + (table_content_width - (table_title_bbox[2] - table_title_bbox[0])) // 2
table_title_y = table_y - 50
draw.text((table_title_x, table_title_y), table_title, fill="black", font=font_title)

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
end_point_x =  cell_x
# 绘制列线，特别是第一列的右边线
for k in range(len(headers) + 1):
    cell_x = table_x + sum(col_widths[:k])
    draw.line([(cell_x, table_y), (cell_x, row_y)], fill="black", width=2)

# 确保第一列右边的线只绘制一次，不重叠
draw.line([(table_x + col_widths[0], table_y), (table_x + col_widths[0], row_y)], fill="black", width=2)

# 右半部分放置人体图片
def black_to_transparent(image_path, output_path):
    """
    替换图像中的黑色背景为透明背景
    """
    # 打开图像
    image = Image.open(image_path).convert("RGBA")
    
    # 获取图像数据
    data = image.getdata()
    
    new_data = []
    for item in data:
        # 替换黑色背景为透明
        if item[:3] == (0, 0, 0):
            new_data.append((0, 0, 0, 0))  # 透明
        else:
            new_data.append(item)
    
    # 更新图像数据
    image.putdata(new_data)
    
    # 保存新的图像
    image.save(output_path, "PNG")


# Function to crop the image based on non-black pixels
def process_image_and_keypoints(skeleton_img, kpts, target_width=300):
    # Step 1: Crop the image to remove black borders
    # def crop_to_content(kpts):
    #     # 找到关键点的最小外接矩形
    #     x_min, y_min = np.min(kpts, axis=0)
    #     x_max, y_max = np.max(kpts, axis=0)
    #     # non_black_mask = np.any(image_array[:, :, :3] != 0, axis=2)
    #     # coords = np.argwhere(non_black_mask)
    #     # y0, x0 = coords.min(axis=0)
    #     # y1, x1 = coords.max(axis=0) + 1  # Add 1 to include the last pixel
    #     return x_min, y_min, x_max, y_max
    x_min, y_min = np.min(kpts, axis=0)
    x_max, y_max = np.max(kpts, axis=0)
    # 裁剪图像，去除黑色背景
    cropped_img = skeleton_img[int(y_min):int(y_max), int(x_min):int(x_max)]

    # 计算宽度的缩放比例，预设宽度为 300
    preset_width = 300
    scale = preset_width / cropped_img.shape[1]

    # 缩放图像到预设宽度
    new_height = int(cropped_img.shape[0] * scale)
    resized_img = cv2.resize(cropped_img, (preset_width, new_height))

    # 调整关键点坐标，根据裁剪和缩放比例
    scaled_kpts = (kpts - [x_min, y_min]) * scale



    return resized_img, scaled_kpts


skeleton_img_path = "demo/resources/skeleton_org.png"
body_image_path = "demo/resources/skeleton.png"
skeleton_img = cv2.imread(skeleton_img_path)
full_kpts = np.load('demo/resources/full_kpts.npy')
target_width = 300
skeleton_img, kpts = process_image_and_keypoints(skeleton_img, full_kpts, target_width)

cv2.imwrite(body_image_path, skeleton_img)
black_to_transparent(body_image_path, body_image_path)
if sex == "男":
    body_image_path = "demo/resources/skeleton.png"
else:
    body_image_path = "demo/resources/skeleton.png"





print("体态问题展示")
grouped_df = combined_df.loc[combined_df.groupby(['class', 'id'])['degree'].idxmax()]
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'degree'].idxmin()], inplace=True)
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左肱骨位置', '右肱骨位置']),'degree'].idxmin()], inplace=True)
# grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左膝超伸', '右膝超伸']),'degree'].idxmin()], inplace=True)
# grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左重心前移', '右重心前移']),'degree'].idxmin()], inplace=True)
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['头部倾斜', '头部水平']),'degree'].idxmin()], inplace=True)
grouped_df.drop([grouped_df.loc[grouped_df['id'].isin(['左肱骨弯曲', '右肱骨弯曲']),'degree'].idxmin()], inplace=True)
grouped_df.loc[grouped_df['id'].isin(['左肱骨位置', '右肱骨位置']),'id'] = '圆肩'
grouped_df.loc[grouped_df['id'].isin(['左肱骨弯曲', '右肱骨弯曲']),'id'] = '肱骨前移'
grouped_df.loc[grouped_df['id']=='头部倾斜','id'] = '头倾斜'
grouped_df.loc[grouped_df['id']=='颈椎倾斜','id'] = '头前引'
grouped_df.loc[grouped_df['id']=='双肩水平','id'] = '高低肩'
oren = grouped_df.loc[grouped_df['id']=='骨盆前/后倾','oren']
grouped_df.loc[grouped_df['id']=='骨盆前/后倾','id'] = '骨盆{}倾'.format(oren.values[0])
oren = grouped_df.loc[grouped_df['id']=='骨盆前/后移','oren']
grouped_df.loc[grouped_df['id']=='骨盆前/后移','id'] = '骨盆{}移'.format(oren.values[0])
grouped_df.loc[grouped_df['id']=='骨盆倾斜','id'] = '骨盆侧倾'
grouped_df = grouped_df[~grouped_df['id'].isin(['膝关节水平', '踝关节水平'])]
oren = grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'oren']
if oren.values[0] == '内':
    grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'id'] = 'X型腿'
else:
    grouped_df.loc[grouped_df['id'].isin(['左腿Q角', '右腿Q角']),'id'] = 'O型腿'
oren = grouped_df.loc[grouped_df['id']=='左足内/外翻','oren']
grouped_df.loc[grouped_df['id']=='左足内/外翻','id'] = '左足{}翻'.format(oren.values[0])
oren = grouped_df.loc[grouped_df['id']=='右足内/外翻','oren']
grouped_df.loc[grouped_df['id']=='右足内/外翻','id'] = '右足{}翻'.format(oren.values[0])
# 将严重程度转换为风险等级
risk_mapping = {
    '正常': '无风险',
    '轻微': '低风险',
    '明显': '中风险',
    '严重': '高风险'
}
grouped_df['risk_level'] = grouped_df['level'].map(risk_mapping)
order = ['头颈部', '肩部', '躯干', '骨盆', '腿部', '脚部']
grouped_df['order'] = grouped_df['class'].map({name: i for i, name in enumerate(order)})
grouped_df = grouped_df.sort_values('order').drop('order', axis=1)

# 获取评估内容和转换后的风险等级
issues = {}
for index, row in grouped_df.iterrows():
    issue = row['id']
    risk_level = row['risk_level']
    issues[f'issue_{index}'] = (risk_level, issue)
# 选择需要展示的列
columns_to_display = ['class', 'id', 'degree', 'oren', 'level', 'risk_level']
df.to_csv('report/full.csv')
df = grouped_df[columns_to_display]
df['degree'] = df['degree'].round(1)  # 保留一位小数

text_x = table_x + sum(col_widths)
text_y = name_y + 50
body_image = Image.open(body_image_path).convert("RGBA")

comprehensive_title = "体态问题概览"
comprehensive_title_bbox = draw.textbbox((0, 0), comprehensive_title, font=font_title)
comprehensive_title_x = text_x + (canvas_width - text_x - (comprehensive_title_bbox[2] - comprehensive_title_bbox[0])) // 2
comprehensive_title_y = text_y
draw.text((comprehensive_title_x, comprehensive_title_y), comprehensive_title, fill="black", font=font_title)

# body_title = "体态综合概览"
# draw.text((text_x, text_y), body_title, fill="black", font=font_title)
################
total_problems = df[df['level'] != '正常'].shape[0]
serious_problems = df[df['risk_level'] == '高风险']['id'].tolist()

# 构造总结文本
summary_text = f"测出体态问题{total_problems}项，其中较为严重的是{', '.join(serious_problems)}。"

# 设置字体
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=25)
font_bold = ImageFont.truetype("demo/SIMYOU.TTF", size=25)  # 假设字体路径中没有明确加粗字体，也可以直接使用原字体

# # 确定起始位置
# text_x = body_image_x  # 假设体态综合概览标题下方
# text_y = body_image_y + body_image.height + 50  # 假设图片下方50px

# 绘制总结文本
text_x += 100
text_y += 50 
draw.text((text_x, text_y), "测出体态问题", fill="black", font=font_text)
text_x += draw.textbbox((0, 0), "测出体态问题", font=font_text)[2]

# 绘制第一个橘黄色加粗字体
problem_count_text = str(total_problems)
draw.text((text_x, text_y), problem_count_text, fill=(255, 165, 0), font=font_bold)
text_x += draw.textbbox((0, 0), problem_count_text, font=font_bold)[2]

remaining_text = summary_text.split('项，其中较为严重的是')
draw.text((text_x, text_y), "项，其中较为严重的是", fill="black", font=font_text)
text_x += draw.textbbox((0, 0), "项，其中较为严重的是", font=font_text)[2]

# 绘制第二个橘黄色加粗字体
serious_problems_text = ', '.join(serious_problems)
draw.text((text_x, text_y), serious_problems_text, fill=(255, 165, 0), font=font_bold)
text_x += draw.textbbox((0, 0), serious_problems_text, font=font_bold)[2]

# 绘制剩余的文本
draw.text((text_x, text_y), "。", fill="black", font=font_text)

font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=25)

####################绘制图片
text_x = table_x + sum(col_widths)
# text_y = name_y + 50
# 调整人体图片大小，保持透明背景
body_image_ratio = 0.5
body_image = body_image.resize((int(body_image.width * body_image_ratio), int(body_image.height * body_image_ratio)), Image.LANCZOS)
body_image_x = text_x + (canvas_width // 2 - body_image.width) // 2

body_image_y = text_y + int(row_height *2.5)
canvas.paste(body_image, (body_image_x, body_image_y), body_image)

# 画布设置
start_x = body_image_x - body_image.width - 100
start_y = body_image_y - 50
column_spacing = body_image.width * 2 + 200  # 两列之间的间隔
row_spacing = 20  # 每个方框之间的垂直间隔
box_width = 150  # 每个方框的宽度
box_height = 70  # 每个方框的高度

# 计算每列可以放置的方框数量
num_rows = (body_image.height - start_y) // (box_height + row_spacing)
num_columns = 2

# 背景颜色变量
background_color = (173, 216, 230)  # 浅蓝色背景

# 风险等级到颜色的映射
color_mapping = {
    '无风险': (0, 128, 0),      # 绿色
    '低风险': (255, 255, 0),    # 浅黄色
    '中风险': (255, 165, 0),    # 橘黄色
    '高风险': (255, 0, 0)       # 红色
}

x17, y17 = body_image_x + body_image_ratio* kpts[[17],0], body_image_y + body_image_ratio* kpts[[17],1] #head
x18, y18 = body_image_x + body_image_ratio* kpts[[18],0], body_image_y + body_image_ratio* kpts[[18],1] #neck
x19, y19 = body_image_x + body_image_ratio* kpts[[19],0], body_image_y + body_image_ratio* kpts[[19],1] #neck
x5, y5 = body_image_x + body_image_ratio* kpts[[5],0], body_image_y + body_image_ratio* kpts[[5],1] #shoulder_right
x6, y6 = body_image_x + body_image_ratio* kpts[[6],0], body_image_y + body_image_ratio* kpts[[6],1] #shoulder_left
x9, y9 = body_image_x + body_image_ratio* kpts[[9],0], body_image_y + body_image_ratio* kpts[[9],1] #hand_right
x10, y10 = body_image_x + body_image_ratio* kpts[[10],0], body_image_y + body_image_ratio* kpts[[10],1] #hand_left
x11, y11 = body_image_x + body_image_ratio* kpts[[11],0], body_image_y + body_image_ratio* kpts[[11],1] #hip_right
x12, y12 = body_image_x + body_image_ratio* kpts[[12],0], body_image_y + body_image_ratio* kpts[[12],1] #hip_left
x9, y9 = body_image_x + body_image_ratio* kpts[[9],0], body_image_y + body_image_ratio* kpts[[9],1]
x10, y10 = body_image_x + body_image_ratio* kpts[[10],0], body_image_y + body_image_ratio* kpts[[10],1]
x14, y14 = body_image_x + body_image_ratio* kpts[[14],0], body_image_y + body_image_ratio* kpts[[14],1]
x13, y13 = body_image_x + body_image_ratio* kpts[[13],0], body_image_y + body_image_ratio* kpts[[13],1]
x16, y16 = body_image_x + body_image_ratio* kpts[[16],0], body_image_y + body_image_ratio* kpts[[16],1]
x15, y15 = body_image_x + body_image_ratio* kpts[[15],0], body_image_y + body_image_ratio* kpts[[15],1]


# 示例数组，实际应用中请根据你的需求定义合适的坐标
start_points = [
    (x17, y17),
    (x18, y18),
    (x6, y6),
    (x5, y5),
    (x6, y6),
    (x19, y18+(y19-y18)//2),
    (x19, y19),
    (x11, y11),
    (x12, y12),
    (x11, y11),
    (x14, y14),
    (x13, y13),
    (x16, y16),
    (x15, y15),
]

cv2_image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGR)

for idx, (index, row) in enumerate(df.iterrows()):
    col = idx % num_columns
    row_pos = idx // num_columns

    # 计算方框位置
    box_x = start_x + col * column_spacing
    box_y = start_y + row_pos * (box_height + row_spacing)

    # 设置文本框背景
    background_box = [box_x, box_y, box_x + box_width, box_y + box_height]
    draw.rectangle(background_box, fill=background_color)

    # 获取风险情况颜色
    text_color = color_mapping.get(row['risk_level'], (0, 0, 0))  # 默认黑色

    # 绘制文本内容（略）

    # 获取起始点坐标
    start_point = start_points[idx]

    # 计算终点坐标
    if col == 0:  # 左列
        end_point = (box_x + box_width, box_y + box_height // 2)
    else:  # 右列
        end_point = (box_x, box_y + box_height // 2)

    # 计算折线的转折点
    bend_x = (start_point[0] + end_point[0]) // 2
    bend_y = start_point[1]

    # 确保坐标为整数类型
    start_point_int = tuple(map(int, start_point))
    end_point_int = (int(bend_x), int(bend_y))

    # 绘制抗锯齿折线
    cv2.line(cv2_image, start_point_int, (int(bend_x), int(bend_y)), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(cv2_image, (int(bend_x), int(bend_y)), tuple(map(int, end_point)), (0, 0, 0), 1, cv2.LINE_AA)

# 将OpenCV图像转换回PIL以继续绘制
canvas = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGBA))
draw = ImageDraw.Draw(canvas)

# 绘制方框和文字
for idx, (index, row) in enumerate(df.iterrows()):
    col = idx % num_columns
    row_pos = idx // num_columns

    # 计算方框位置
    box_x = start_x + col * column_spacing
    box_y = start_y + row_pos * (box_height + row_spacing)

    # 设置文本框背景
    background_box = [box_x, box_y, box_x + box_width, box_y + box_height]
    draw.rectangle(background_box, fill=background_color)

    # 获取风险情况颜色
    text_color = color_mapping.get(row['risk_level'], (0, 0, 0))  # 默认黑色

    # 计算风险情况和体态问题的文本尺寸
    level_text = row['risk_level']
    issue_text = row['id']

    # 使用 textbbox 计算文本的边界框，然后计算宽度和高度
    level_bbox = draw.textbbox((0, 0), level_text, font=font_text)
    level_width = level_bbox[2] - level_bbox[0]
    level_height = level_bbox[3] - level_bbox[1]

    issue_bbox = draw.textbbox((0, 0), issue_text, font=font_text)
    issue_width = issue_bbox[2] - issue_bbox[0]
    issue_height = issue_bbox[3] - issue_bbox[1]

    # 计算文字位置，居中显示
    level_x = box_x + (box_width - level_width) // 2
    level_y = box_y + 5  # 第一行，靠上

    issue_x = box_x + (box_width - issue_width) // 2
    issue_y = box_y + box_height // 2  # 第二行，居中偏下

    # 绘制风险情况（第一行）
    draw.text((level_x, level_y), level_text, font=font_text, fill=text_color)

    # 绘制体态问题（第二行）
    draw.text((issue_x, issue_y), issue_text, font=font_text, fill=(0, 0, 0))  # 黑色字体

start_x = table_x + sum(col_widths) + 100
start_y = box_y +100 + int(row_height*1.5)
note = '备注:本报告仅对检测图像信息负责、因衣服、姿态、拍照\n角度等影响可能会造成误差，不作为医学诊断依据。'
font_text = ImageFont.truetype("demo/SIMYOU.TTF", size=25)

# 在画布上绘制文本
current_y = start_y
lines = note.split('\n')  # 按照 '\n' 进行手动换行
for line in lines:
    draw.text((start_x, current_y), line, font=font_text, fill="black")
    # 使用 textbbox 计算下一行的 Y 坐标
    text_bbox = draw.textbbox((0, 0), line, font=font_text)
    line_height = text_bbox[3] - text_bbox[1]
    current_y += line_height + 5  # 行间距设为 5px
# 保存最终图像
canvas.save("demo/reports/posture_evaluation_report.png", format="PNG")
