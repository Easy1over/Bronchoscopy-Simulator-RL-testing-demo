import vtk
import numpy as np
import csv
from vtkmodules.util import numpy_support
from sklearn.neighbors import KDTree

#vtk转obj文件
def vtk_to_obj(input_vtk_file, output_obj_file):
    # 1. 读取VTK文件
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(input_vtk_file)
    reader.Update()

    # 2. 将VTK数据转换为PolyData，以确保适配OBJ格式
    if not isinstance(reader.GetOutput(), vtk.vtkPolyData):
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputConnection(reader.GetOutputPort())
        geometry_filter.Update()
        poly_data = geometry_filter.GetOutput()
    else:
        poly_data = reader.GetOutput()

    # 3. 创建OBJ writer
    obj_writer = vtk.vtkOBJWriter()
    obj_writer.SetInputData(poly_data)
    obj_writer.SetFileName(output_obj_file)

    # 4. 写出OBJ文件
    obj_writer.Write()

#obj转列表
def read_obj_points(obj_file_path):
    points_list = []

    with open(obj_file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v'):  # 检查是否为顶点定义行（以'v'开头）
                coordinates = line.split()[1:]  # 分割并忽略第一个字符（'v'）
                x, y, z = [float(coord) for coord in coordinates]  # 将坐标字符串转换为浮点数
                points_list.append([x, y, z])  # 将xyz坐标添加到列表

    return points_list

# 逆转列表+四舍五入 pointlist代表四舍五入的位数
def round_reverse_list(my_list,pointlen):
    anslist = []
    for point in my_list:
        rounded_point = [round(coord,pointlen) for coord in point]
        anslist.append(rounded_point)
    list_result=anslist[::-1]
    return list_result

#输入一个列表，返回一个列表，全部取相反数 这个主要是用于反平移回坐标原点用的
def negative_list(list):
    my_negative_list=[]
    for point in list:
        point1=-point
        my_negative_list.append(point1)
    return my_negative_list

#长列表转短列表
def longtoshort_list(list,target_len):
    len0=len(list)
    interval = len0 // (target_len - 1)
    sub_list = [list[0]] + [list[i * interval] for i in range(1, target_len - 1)]
    return sub_list

#列表存入 一个csv文件
def write_csv(ofile,my_list):
    saved_file=f'{ofile}\points_left.csv'
    with open(saved_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point in my_list:
            writer.writerow(point)

#读取 csv文件 变成一个列表
def read_csv(filename):
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        point_list = []
        for row in reader:
            point_list.append([float(coord) for coord in row])
        return point_list

# 将中心线平移到原点附近，得到新的列表 
# 需要如输入的是：初始点列表，平移量（也就是这个初始中心线第一个点的相反数，每个点平移这么多后，起点就到原点去了，再进行旋转等操作）
# 返回新列表
def CLtoOrigin(my_list,tran_list):
    Cl0_array=np.array(my_list)
    translation_vector = np.array(tran_list)
    translated_points = Cl0_array + translation_vector
    #rotation_axis = np.array([0, 1, 0])
    angle_rad = np.radians(90)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_theta, 0, -sin_theta],
                            [0, 1, 0],
                            [sin_theta, 0, cos_theta]])
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    new_point_cloud = rotated_points.tolist()
    #为了避免模型碰撞，这里还向x正方向平移了100
    new_point_cloud = new_point_cloud+ np.array([100,0,0])
    return new_point_cloud

#输入一个三维数组，输出一个vtk能使用的格式：imagedata
def np_array2vtk_image(model):
    np_array=model

# vtkImageData中的数据全部都平铺成了一维数组,所以此处使用ravel()函数进行平铺处理
    depth_arr = numpy_support.numpy_to_vtk(np_array.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    im_data = vtk.vtkImageData()
	

    if np_array.ndim == 3:
        # 确保维度顺序为 (x, y, z) - VTK 中通常是 (x, y, z)
        np_array = np.transpose(np_array, (2, 1, 0))  # 适应 VTK 的坐标系统   

    im_data.SetDimensions(np_array.shape)
    #设置像素大小
    size=0.1
    im_data.SetSpacing([size, size, size])
    #设置初始位置
    im_data.SetOrigin([0, 0, 0])
	# 设置数据信息
    im_data.GetPointData().SetScalars(depth_arr)

    return im_data 

#这个函数输入一个模型列表和它的长度，输出一个一维列表，每个元素是一个vtkimage的模型
def Allmodel_Vtkimagemodel(all_model,len):
    all_vtkimage_model=[]
    for i in range(len):
        vtkimage_model=np_array2vtk_image(all_model[i])
        all_vtkimage_model.append(vtkimage_model)
    return all_vtkimage_model

#下面这些是手柄计算的函数
def Solve_theta(x,y):
    theta=np.arctan2(y,x)
    theta=int(theta*180/np.pi)
    return theta

def Flag_curve(x,y):
    flag=int(np.trunc(40*np.sqrt(x*x+y*y)))
    return flag

def Flag_forward(forward):
    if np.abs(forward)<0.0:
        xx=0
    else:
        xx=(-(np.round((np.abs(forward)-0.0)*10,3))*np.sign(forward))/10
    return xx

    #右乘：自身坐标系 如下的先执行旋转再执行位移(注意顺序)

def My_translate(theta,X):
    New_Transform=vtk.vtkTransform()        
    New_Transform.PostMultiply()
    New_Transform.Translate (-40+X,-1.5,-30)
    New_Transform.RotateX(theta)
    return New_Transform


# 读入vtk文件转列表
def read_vtk_to_list(file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
 
    points = reader.GetOutput().GetPoints()
    num_points = points.GetNumberOfPoints()
 
    point_list = []
    for i in range(num_points):
        point = [0.0, 0.0, 0.0]
        points.GetPoint(i, point)
        point_list.append(point)
 
    return point_list



# 1. 加载点数据
def load_points_from_file(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f.readlines(), start=1):
            line = line.strip()  # 去除每行的空白字符
            if line:
                try:
                    point = list(map(float, line.split(',')))  # 将每个点转换为浮动数
                    if len(point) == 3:  # 确保是三维点
                        points.append(point)
                    else:
                        print(f"Skipping invalid point in line {line_number}: {point}")
                except ValueError:
                    print(f"Error parsing line {line_number}: {line}")
    return np.array(points, dtype=np.float32)

# 2. 使用 KDTree 查找最近邻
def find_closest_point(point, points):
    # 使用 KDTree 加速最近邻查询
    input_point=np.array(point, dtype=np.float32)
    tree = KDTree(points)
    dist, ind = tree.query([input_point], k=1)  # 查找最近的一个点
    return points[ind[0][0]]  # 返回最近的点

#向量归一化，格式要求[1，2，3]
def normalize_vector(vector):
    """
    归一化向量
    :param vector: 三维向量 [x, y, z]
    :return: 归一化后的向量
    """
    # 使用 numpy 计算向量的模长
    magnitude = np.linalg.norm(vector)
    
    #if magnitude == 0:
    #    raise ValueError("Cannot normalize the zero vector")
    
    # 归一化向量
    normalized_vector = vector / magnitude
    return normalized_vector

def Transform_WorldtoTrail(target_position, target_direction):
    """
    计算从初始状态到目标状态的变换矩阵
    :param target_position: 目标位置 [a, b, c]
    :param target_direction: 目标方向 [x, y, z]（单位向量）
    :return: 变换矩阵 T
    """
    # 初始方向和位置
    initial_position = np.array([0, 0, 0])
    initial_direction = np.array([1, 0, 0])  # 初始方向向量

    # 归一化目标方向向量

    target_direction = np.array(target_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)

    # 计算旋转轴（叉积）和旋转角度（夹角）
    axis = np.cross(initial_direction, target_direction)
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)  # 归一化旋转轴

        cos_theta = np.dot(initial_direction, target_direction)
        theta = np.arccos(cos_theta)  # 旋转角度
        # Rodrigues' 旋转公式
        K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])

        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # 如果旋转轴为零，表示两个向量是共线的，不需要旋转
    elif np.linalg.norm(axis) == 0:
        # return np.eye(4)  # 返回单位矩阵（不需要旋转）
        R=np.eye(3)

    # 目标位置作为平移向量

    t = np.array(target_position)-initial_position  # 目标位置 [x, y, z]

    # 创建4x4的变换矩阵
    T = np.eye(4)  # 单位矩阵
    T[:3, :3] = R  # 将旋转矩阵放入左上角
    T[:3, 3] = t  # 将平移向量放入左上角

    return T

def Transform_TrailtoHand(theta,phi):
    #角度转弧度
    theta=np.radians(theta)
    phi=np.radians(phi)

    #np.cos(theta) np.sin(theta) np.cos(phi) np.sin(phi)
    l=35
    # 1. 创建平移矩阵
    if phi != 0:
        T_translation = np.array([[1, 0, 0,l/phi*np.sin(phi)],
                                [0, 1, 0, l/phi*np.sin(theta)*(np.cos(phi)-1)],
                                [0, 0, 1, -l/phi*np.cos(theta)*(np.cos(phi)-1)],
                                [0, 0, 0, 1]])
    else:
        T_translation = np.array([[1, 0, 0,l],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    # 2. 创建绕x轴旋转矩阵
    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])

    # 3. 创建绕y轴旋转矩阵
    R_y = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    transformation_matrix = np.dot(T_translation, np.dot(R_x, R_y))

    '''
    R=np.array([[np.cos(theta), -np.cos(phi)*np.sin(theta), -np.sin(theta)*np.sin(phi)],   # 假设是绕Z轴旋转90度
                [np.sin(theta),  np.cos(theta)*np.cos(phi), np.sin(phi)*np.cos(theta)], 
                [0,  np.sin(phi), np.cos(phi)]])
    l=35
    if phi != 0:
        #t=np.array([l/phi*np.sin(theta)*(np.cos(phi)-1), -l/phi*np.cos(theta)*(np.cos(phi)-1), l/phi*np.sin(phi)])
        t=np.array([l/phi*np.sin(phi), l/phi*np.sin(theta)*(np.cos(phi)-1),-l/phi*np.cos(theta)*(np.cos(phi)-1)])
    if phi == 0:
        t=np.array([l, 0, 0])
    '''
    # 创建 4x4 单位矩阵
    #transformation_matrix = np.eye(4)
    
    # 将旋转矩阵插入到变换矩阵的左上角 (3x3)
    #transformation_matrix[:3, :3] = R
    #transformation_matrix[:3, :3] = R

    # 将平移向量插入到变换矩阵的最后一列 (3x1)
    #transformation_matrix[:3, 3] = t

    return transformation_matrix

def apply_transform_to_polydata(polydata, transform):
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()

def check_collision(robot_actor, airway_actor):
    # 获取变换后的机器人和支气管树表面
    robot_surface = apply_transform_to_polydata(robot_actor.GetMapper().GetInput(), robot_actor.GetUserTransform())
    airway_surface = apply_transform_to_polydata(airway_actor.GetMapper().GetInput(), airway_actor.GetUserTransform())

    # 使用 vtkOBBTree 进行碰撞检测
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(airway_surface)
    obb_tree.BuildLocator()

    print(f"Number of points in robot surface: {robot_surface.GetNumberOfPoints()}")
    print(f"Number of lines in robot surface: {robot_surface.GetNumberOfLines()}")

    if robot_surface.GetNumberOfLines() == 0:
        print("Warning: No lines found in robot surface. Ensure vtkTriangleFilter has been applied correctly.")


    # 遍历机器人的所有线段，检查是否与支气管树相交
    points = vtk.vtkPoints()
    cell_lines = robot_surface.GetLines()
    cell_lines.InitTraversal()
    id_list = vtk.vtkIdList()

    while cell_lines.GetNextCell(id_list):
        for i in range(id_list.GetNumberOfIds() - 1):
            pt1 = robot_surface.GetPoint(id_list.GetId(i))
            pt2 = robot_surface.GetPoint(id_list.GetId(i + 1))
            code = obb_tree.IntersectWithLine(pt1, pt2, points, None)
            if code != 0:
                return True

    return False

def resample_volume(image_data, scale_factor):
    """
    使用 vtkImageResample 来调整体积数据的分辨率。

    Args:
        image_data (vtk.vtkImageData): 输入的体积数据。
        scale_factor (float): 缩放因子，决定分辨率的降低程度。

    Returns:
        vtk.vtkImageData: 调整分辨率后的体积数据。
    """
    # 创建 vtkImageResample 对象
    resample = vtk.vtkImageResample()
    resample.SetInputData(image_data)

    # 设置缩放因子
    resample.SetAxisMagnificationFactor(0, scale_factor)  # 设置X轴的缩放比例
    resample.SetAxisMagnificationFactor(1, scale_factor)  # 设置Y轴的缩放比例
    resample.SetAxisMagnificationFactor(2, scale_factor)  # 设置Z轴的缩放比例

    # 执行重采样
    resample.Update()

    # 获取并返回重采样后的数据
    return resample.GetOutput()

def visualize_vtk_voxel_model(vti_filename):
    """
    可视化一个 VTK 格式的体素化模型（.vti 文件），使用体积渲染。

    参数:
        vti_filename (str): VTK 体素化模型文件的路径（.vti 格式）。
    """
    # 创建一个读取器来读取 VTI 文件
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_filename)
    reader.Update()  # 读取文件内容
    
    # 获取读取的数据
    image_data = reader.GetOutput()
    scale_factor = 1  # 将体积数据分辨率缩小为原来的50%
    resampled_image_data = resample_volume(image_data, scale_factor)
    # 创建映射器，使用体积渲染
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(resampled_image_data)
    
    # 设置体积的颜色和不透明度
    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)  # 空白处颜色（黑色）
    volume_color.AddRGBPoint(1.0, 1.0, 1.0, 1.0)  # 气管处颜色（白色）
    
    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0.0, 0.0)  # 空白处透明
    volume_scalar_opacity.AddPoint(1.0, 1.0)  # 气管部分不透明
    
    # 创建体积属性
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)  # 设置颜色
    volume_property.SetScalarOpacity(volume_scalar_opacity)  # 设置透明度
    volume_property.ShadeOn()  # 开启阴影效果
    
    # 创建体积对象
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

def load_mapper_list(base_filename, num_mappers):
    mapper_list = []
    for i in range(num_mappers):
        # 生成文件名
        #filename = f"{base_filename}_{i}.vtk"
        #filename = f"{base_filename}_{i}.vtp"
        filename = f"{base_filename}_{i}.stl"
        
        # 读取 vtkPolyData
        #reader = vtk.vtkPolyDataReader()
        #reader = vtk.vtkXMLPolyDataReader()
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()  # 读取数据
        
        # 获取 polydata
        polydata = reader.GetOutput()
        
        # 创建一个 PolyDataMapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        # 将mapper添加到列表中
        mapper_list.append(mapper)
        print(f"Loaded {filename}")
    
    return mapper_list

def vector_rotateYX(vector):
#这个函数用于计算一个向量等效于绕Y轴旋转theta角度后再绕X轴旋转phi角度对应的角度值
    X,Y,Z=vector[0],vector[1],vector[2]
    R=np.sqrt(Y**2+Z**2)
    thetaY=np.degrees(np.arctan2(R,X))
    thetaX=np.degrees(np.arctan2(Z,Y))+90
    return thetaY,thetaX

def check_point_collision(point, polydata):
    """
    检查一个点是否与 STL 模型的顶点发生碰撞
    :param point: 需要检测的点，格式为 (x, y, z)
    :param polydata: STL 模型的 polydata 对象
    :return: 如果碰撞返回 True，否则返回 False
    """
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    for i in range(num_points):
        env_point = points.GetPoint(i)
        dx = point[0] - env_point[0]
        dy = point[1] - env_point[1]
        dz = point[2] - env_point[2]
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if distance < 0.1:  # 假设碰撞阈值为 0.1
            return True
    return False

def read_csv_to_vectors(file_path):
    vectors = []  # 用于保存三维向量的列表

    # 打开 CSV 文件并读取数据
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)

        # 遍历文件的每一行
        for row in reader:
            # 假设每一行都有 3 个数，分别对应 x, y, z
            if len(row) == 3:
                try:
                    # 将每行数据转换为浮点数并组成一个三维向量
                    vector = [float(row[0]), float(row[1]), float(row[2])]
                    vectors.append(vector)  # 将三维向量加入列表
                except ValueError:
                    # 如果转换为浮点数失败，跳过该行
                    print(f"Skipping invalid row: {row}")
            else:
                print(f"Skipping row with incorrect number of elements: {row}")

    return vectors

# 读取CSV文件并返回坐标列表的函数
def read_csv(file_path):
    points = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 假设每行的格式是: [x, y, z]
            points.append([float(coord) for coord in row])
    return points

#拼接列表
def get_concatenated_points(files):
    all_points = []
    for file in files:
        points = read_csv(file)
        all_points.extend(points)  # 拼接到同一个列表
    return np.array(all_points)

# 计算任务完成度
def calculate_completion(idx, total_points):
    return idx / total_points  # 最近点的索引除以总点数，得到比值

# 计算头部点与点列表中最近的点，并返回其顺序编号
def find_nearest_point(point, points):
    # 使用 KDTree 加速最近邻查询
    input_point=np.array(point, dtype=np.float32)
    tree = KDTree(points)
    dist, ind = tree.query([input_point], k=1)  # 查找最近的一个点
    return ind,points[ind[0][0]]  # 返回最近的点


def read_points_from_csv(file_name):
    """从CSV文件中读取点数据"""
    points = vtk.vtkPoints()
    counter = 0  # 计数器用于控制读取间隔
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        #next(reader)  # 跳过表头行，如果有的话
        for row in reader:
            if counter % 8 == 0:  # 每隔5个点处理一次
            #if 1:
                x, y, z = map(float, row[0:3])  # 假设每行数据格式为 [时间, X, Y, Z]
                x /= 10.0
                y /= 10.0
                z /= 10.0
                points.InsertNextPoint(x, y, z)
            counter += 1
    return points


def create_polyline(points, threshold=10):
    """根据给定的点和阈值创建Polyline，两点间距离大于阈值时不连线"""
    polydata = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()

    num_points = points.GetNumberOfPoints()
    if num_points < 2:
        return polydata  # 如果点少于两个，直接返回空的polydata

    currentLine = vtk.vtkPolyLine()
    currentLine.GetPointIds().Allocate(2)

    previousPointId = 0
    currentLine.GetPointIds().InsertNextId(previousPointId)

    for i in range(1, num_points):
        point = np.array(points.GetPoint(i))
        prev_point = np.array(points.GetPoint(previousPointId))
        
        distance = np.linalg.norm(point - prev_point)
        
        if distance <= threshold:
            currentLine.GetPointIds().InsertNextId(i)
        else:
            # 结束当前线段并添加到lines
            lines.InsertNextCell(currentLine)
            
            # 开始新的线段
            currentLine = vtk.vtkPolyLine()
            currentLine.GetPointIds().Allocate(2)
            currentLine.GetPointIds().InsertNextId(i)
        
        previousPointId = i
    
    # 插入最后一个polyline
    if currentLine.GetPointIds().GetNumberOfIds() > 0:
        lines.InsertNextCell(currentLine)
    
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def read_last_point_from_csv(csv_path: str):
    """
    读取CSV文件最后一个有效坐标点，返回 [x1, x2, x3]（float）。
    - 跳过空行
    - 自动跳过表头/非数字行
    - 默认取每行前3列作为 x,y,z
    """
    last_point = None

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # 去掉空白
            row = [c.strip() for c in row if c is not None]

            # 至少要有3列
            if len(row) < 3:
                continue

            try:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
                last_point = [x, y, z]
            except ValueError:
                # 可能是表头或非数字行，跳过
                continue

    if last_point is None:
        raise ValueError(f"CSV中未找到任何有效的三维坐标行：{csv_path}")

    return last_point
