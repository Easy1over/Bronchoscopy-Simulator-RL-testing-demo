import time
import numpy as np
import My_Funs as mf
import vtk
import csv
from scipy.spatial import KDTree
import random
import pickle
import cv2
from filelock import FileLock

# ===== 全局变量 =====
abs_y_state = 0
abs_rx_state = 0
abs_ry_state = 0
flag_curve = 0
flag_forward = 0
trajectory_points = []
global smooth_rx, smooth_ry
smooth_rx = 0.0
smooth_ry = 0.0

#target_right = [133.029, 18.2965, -71.8174]
#target_left = [144.3673, 47.0943, 56.6057]

Tree_Point_file = rf'E:\Project\VSCProject\Bronchoscopy\Code_Vmtk_Contral\New_models\new_all_cl.csv'
Tree_Point = mf.load_points_from_file(Tree_Point_file)

# Pmappers = mf.load_mapper_list(
#     "E:/Project/VSCProject/Bronchoscopy/Code_Vmtk_Contral/Model/Robotmodel", 41
# )

# ===== CSV工具 =====
def read_csv(file_path):
    points = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            points.append([float(coord) for coord in row])
    return points

def get_concatenated_points(files):
    all_points = []
    for file in files:
        all_points.extend(read_csv(file))
    return np.array(all_points)

# ===== Point 类 =====
class Point:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    def Hand_Point(self, theta, phi):
        T_WorldtoTrail = mf.Transform_WorldtoTrail(self.position, self.direction)
        T_TrailtoHand = mf.Transform_TrailtoHand(theta, phi)
        T_WorldtoHand = np.dot(T_WorldtoTrail, T_TrailtoHand)

        Handposition = T_WorldtoHand[:3, 3]
        rotation_matrix = T_WorldtoHand[:3, :3]
        Handdirection = np.dot(rotation_matrix, [1, 0, 0])
        return Handposition, Handdirection

    def Translate_M(self, theta):
        thetaY, thetaX = mf.vector_rotateYX(self.direction)

        New_Transform = vtk.vtkTransform()
        New_Transform.PreMultiply()
        New_Transform.Translate(*self.position)
        New_Transform.RotateX(thetaX)
        New_Transform.RotateY(thetaY)
        New_Transform.RotateX(theta - thetaX)
        New_Transform.Translate(-5, -1.5, -30)
        return New_Transform

# ===== 中心线数据 =====
new_all_cl_path = [
    r"E:\Project\VSCProject\Bronchoscopy\Code_Vmtk_Contral\New_models\new_all_cl.csv"
]
mylinepointsall = get_concatenated_points(new_all_cl_path)

i=1 #挑选路线
target_cl_path = [
    rf"E:\Project\VSCProject\Bronchoscopy\Code_Vmtk_Contral\New_models\cl_CSV\Centerline model_{i}_line1.csv"
]
mylinepoints1 = get_concatenated_points(target_cl_path)
linepointnum = len(mylinepoints1) 

target_csv_path = rf"E:\Project\VSCProject\Bronchoscopy\Code_Vmtk_Contral\New_models\cl_CSV\Centerline model_{i}_line1.csv"
target_point=mf.read_last_point_from_csv(target_csv_path)


# ===== 主程序 =====
def main():


    # ===== VTK 渲染窗口 =====
    global renderer_camera, ren_win, camera
    #global renderer
    #renderer = vtk.vtkRenderer()
    #renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
    #renderer.SetViewport(0.0, 0.0, 1.0, 1.0)

    renderer_camera = vtk.vtkRenderer()
    renderer_camera.SetViewport(0.0, 0.0, 1.0, 1.0)

    ren_win = vtk.vtkRenderWindow()
    #ren_win.AddRenderer(renderer)
    ren_win.AddRenderer(renderer_camera)

    #ren_win.SetSize(1500, 750)
    ren_win.SetSize(1000, 1000)

    #renderer.SetBackground(0.7, 0.7, 0.7)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    # ===== 摄像机 =====
    camera = vtk.vtkCamera()
    camera.SetPosition(0, 0, 0)
    camera.SetFocalPoint(1, 0, 0)
    camera.SetViewUp(0, 1, 0)
    camera.SetClippingRange(0.1, 1000)
    camera.SetViewAngle(80)

    renderer_camera.SetActiveCamera(camera)
    renderer_camera.SetBackground(0.1, 0.1, 0.1)

    # ===== 光源 =====
    global scope_light
    scope_light = vtk.vtkLight()
    scope_light.SetLightTypeToSceneLight()
    scope_light.SetPositional(True)
    scope_light.SetConeAngle(50)
    scope_light.SetIntensity(1.0)
    scope_light.SetAttenuationValues(0.3, 0.0001, 0.0005)
    renderer_camera.AddLight(scope_light)

    camera.SetPosition(0, 0, 0)
    camera.SetFocalPoint(1, 0, 0)

    scope_light.SetPosition(0, 0, 0)
    scope_light.SetFocalPoint(1, 0, 0)

    # ===== 相机箭头 =====
    arrow_source = vtk.vtkArrowSource()
    arrow_mapper = vtk.vtkPolyDataMapper()
    arrow_mapper.SetInputConnection(arrow_source.GetOutputPort())

    arrow_actor = vtk.vtkActor()
    arrow_actor.SetMapper(arrow_mapper)
    arrow_actor.GetProperty().SetColor(1, 0, 0)

    arrow_transform = vtk.vtkTransform()
    arrow_transform.Translate(camera.GetPosition())
    arrow_transform.RotateWXYZ(camera.GetViewAngle(), 0, 1, 0)
    arrow_actor.SetUserTransform(arrow_transform)
    #renderer.AddActor(arrow_actor)

    # ===== 机器人 Actor =====
    global tail_point
    tail_point = Point([0, 0, 0], [1, 0, 0])

    Ori_Transform = vtk.vtkTransform()
    Ori_Transform.PostMultiply()
    Ori_Transform.Translate(-5, -1.5, -30)

    global robot_actor
    robot_actor = vtk.vtkActor()
    #robot_actor.SetMapper(Pmappers[0])
    robot_actor.SetUserTransform(Ori_Transform)
    robot_actor.GetProperty().SetColor(0.4, 0.4, 0.4)
    #renderer.AddActor(robot_actor)

    # ===== 支气管模型 =====
    file_path = (
        "E:/Project/VSCProject/Bronchoscopy/Code_Vmtk_Contral/New_models/"
        "AirwayHollow_siliconmodel3_simUV.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    # 平滑
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(reader.GetOutputPort())
    smooth.SetNumberOfIterations(20)
    smooth.SetPassBand(0.1)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    # =======================================
    # A) 渲染模型：完全保持你原本的 normals（不参与碰撞计算）
    # =======================================
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.SetConsistency(True)
    normals.SplittingOff()
    normals.SetFeatureAngle(85)
    normals.Update()

    render_surface = normals.GetOutput()   # 这个用于显示

    # 加入渲染 actor（保持原样）
    swmodel_mapper = vtk.vtkPolyDataMapper()
    swmodel_mapper.SetInputData(render_surface)

    global swmodel_actor
    swmodel_actor = vtk.vtkActor()
    swmodel_actor.SetMapper(swmodel_mapper)

    swmodel_T = vtk.vtkTransform()
    swmodel_T.PostMultiply()
    swmodel_T.Translate(4.21889, -22.6936, -28.5903)
    swmodel_T.RotateY(270)
    swmodel_actor.SetUserTransform(swmodel_T)

    bronch_prop = swmodel_actor.GetProperty()
    bronch_prop.SetColor(0.88, 0.53, 0.47)
    bronch_prop.SetOpacity(1)
    bronch_prop.SetInterpolationToPhong()

    bronch_prop.SetAmbient(0.08)
    bronch_prop.SetDiffuse(0.90)
    bronch_prop.SetSpecular(0.03)       # 非常低
    bronch_prop.SetSpecularPower(8)     # 模糊

    base_color = [0.88, 0.53, 0.47]
    jitter = 0.02

    bronch_prop.SetColor(
        base_color[0] + random.uniform(-jitter, jitter),
        base_color[1] + random.uniform(-jitter, jitter),
        base_color[2] + random.uniform(-jitter, jitter),
    )


    renderer_camera.AddActor(swmodel_actor)

    # ===================================================
    # B) 计算模型：独立生成，用于碰撞检测（不可见）
    # ===================================================
    # （1）从 smooth 输出重新生成 CellNormals（使用 CellNormals 更稳）
    normalGen_calc = vtk.vtkPolyDataNormals()
    normalGen_calc.SetInputConnection(smooth.GetOutputPort())
    normalGen_calc.ComputePointNormalsOff()
    normalGen_calc.ComputeCellNormalsOn()
    normalGen_calc.SplittingOff()
    normalGen_calc.Update()

    calc_surface = normalGen_calc.GetOutput()

    # （2）应用相同 transform 到计算模型
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(swmodel_T)
    transform_filter.SetInputData(calc_surface)
    transform_filter.Update()

    # 计算模型（用于 OBBTree）
    airway_surface_world = transform_filter.GetOutput()

    # （3）构建 OBBTree（inside/outside 判定用）
    global airway_tree
    airway_tree = vtk.vtkOBBTree()
    airway_tree.SetDataSet(airway_surface_world)
    airway_tree.BuildLocator()
    airway_tree.SetMaxLevel(10)  # 稳定 inside/outside

    # 可视化 OBBTree 包围盒（Debug）
    # ==========================================
    obb_poly = vtk.vtkPolyData()
    airway_tree.GenerateRepresentation(0, obb_poly)

    obb_mapper = vtk.vtkPolyDataMapper()
    obb_mapper.SetInputData(obb_poly)

    # obb_actor = vtk.vtkActor()
    # obb_actor.SetMapper(obb_mapper)
    # obb_actor.GetProperty().SetColor(1, 1, 0)  # 黄色框
    # obb_actor.GetProperty().SetLineWidth(3)
    # obb_actor.GetProperty().SetRepresentationToWireframe()
    # obb_actor.GetProperty().SetOpacity(1.0)

    # 在任意 renderer 上显示（你看得清楚即可）
    global enclosed_checker

    enclosed_checker = vtk.vtkSelectEnclosedPoints()
    enclosed_checker.SetSurfaceData(airway_surface_world)  # 你的支气管表面
    #enclosed_checker.Initialize()   

    # 透明管壁版本
    global swmodel_actor1
    swmodel_actor1 = vtk.vtkActor()
    swmodel_actor1.SetMapper(swmodel_mapper)
    swmodel_actor1.SetUserTransform(swmodel_T)
    swmodel_actor1.GetProperty().SetColor(0.88, 0.53, 0.47)
    swmodel_actor1.GetProperty().SetOpacity(0.3)
    #renderer.AddActor(swmodel_actor1)

    # 目标球
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(5)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.GetProperty().SetColor(0, 1, 0)
    sphere_actor.GetProperty().SetOpacity(0.5)

    sphere_T = vtk.vtkTransform()
    sphere_T.PostMultiply()
    sphere_T.Translate(144.3673, 47.0943, 56.6057)
    sphere_actor.SetUserTransform(sphere_T)

    # ===== 中心线显示 =====
    points1 = mf.read_points_from_csv(
        "E:/Project/VSCProject/Bronchoscopy/Tree/Tree_CL/bronchial_cl2.csv"
    )
    polyline1 = mf.create_polyline(points1, threshold=12)
    line1_mapper = vtk.vtkPolyDataMapper()
    line1_mapper.SetInputData(polyline1)
    line1_actor = vtk.vtkActor()
    line1_actor.SetMapper(line1_mapper) 
    line1_actor.GetProperty().SetColor(1, 0, 0)
    line1_actor.GetProperty().SetLineWidth(3)

    # OBJ中心线
    centerline_reader = vtk.vtkOBJReader()
    centerline_reader.SetFileName(
        "E:/Project/VSCProject/Bronchoscopy/Code_Vmtk_Contral/"
        "New_models/trans_cl/Centerline model_1.obj"
    )
    centerline_reader.Update()

    centerline_mapper = vtk.vtkPolyDataMapper()
    centerline_mapper.SetInputConnection(centerline_reader.GetOutputPort())

    centerline_actor = vtk.vtkActor()
    centerline_actor.SetMapper(centerline_mapper)
    centerline_actor.GetProperty().SetColor(0, 1, 0)
    centerline_actor.GetProperty().SetLineWidth(3)
                                        

    print("Ready to show...")
    time.sleep(0.5)

    #ren_win.SetSize(1500, 750)
    ren_win.Render()

    iren.AddObserver("TimerEvent", timer_callback)
    iren.CreateRepeatingTimer(25)
    iren.Initialize()
    iren.Start()

def is_inside_airway(point):
    global enclosed_checker

    pts = vtk.vtkPoints()
    pts.InsertNextPoint(point)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)

    enclosed_checker.SetInputData(poly)
    enclosed_checker.Update()

    return enclosed_checker.IsInside(0) == 1

def is_outside_airway(point):
    global airway_tree

    p = np.array(point, dtype=float)
    #print(f"Checking point: {p}")
    # VTK 的 inside/outside 检测
    result = airway_tree.InsideOrOutside(p)

    # result = 1 表示点在模型外部
    return result == 1 #1代表外部 0代表内部

def is_valid_bending(rx, ry):
    Theta = mf.Solve_theta(rx, ry)
    Phi   = 2.5 * mf.Flag_curve(rx, ry)

    HandPos, _ = tail_point.Hand_Point(Theta, Phi)

    #return not is_outside_airway(HandPos)
    return not is_inside_airway(HandPos)

def check_render_window(render_window):
    """
    检查渲染窗口是否准备好
    通过获取渲染窗口的图像尺寸来确认渲染窗口是否已经渲染了有效内容
    """
    # 获取渲染窗口的尺寸
    width, height = render_window.GetSize()
    
    # 检查尺寸是否有效（不为0）
    if width > 0 and height > 0:
        #print("good")
        return True
    return False

def capture_image(render_window):

    # 创建一个 vtkWindowToImageFilter 用于从渲染窗口中提取图像
    while True:
        if not check_render_window(render_window):
            print("Render window not ready yet, skipping capture.")
            time.sleep(0.0005)
            print(1)
        else:
            break

    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    # 获取图像数据并转化为 NumPy 数组
    image_data = window_to_image_filter.GetOutput()
    # 将 vtkImageData 转换为 NumPy 数组 (height, width, channels)
    width, height, _ = image_data.GetDimensions()  # 获取图像宽高
    vtk_array = image_data.GetPointData().GetScalars()
    
    # 从 vtk 数组中提取数据并转换为 NumPy 数组
    np_image = np.frombuffer(vtk_array, dtype=np.uint8)
    np_image = np_image.reshape((height, width, 3))  # 转换为 (height, width, channels)

    # 将RGB转为BGR，解决颜色通道问题 OpenCV等库常用BGR格式
    np_image = np_image[:, :, ::-1]  # 切片反转通道顺序

    resized_image = cv2.resize(np_image, (224, 224))  # HWC
    chw_image = np.transpose(resized_image, (2, 0, 1))  # CHW
    normalized = chw_image.astype(np.float32) / 255.0

    return normalized

#定义一个求当前点到目标点位置的函数
def target_to_current(target_position,current_position):
    #读取目标点
    list1 = np.array(target_position)
    list2 = np.array(current_position)
    result=list1-list2
    return np.round(result,3).tolist()

#回调函数
def timer_callback(caller, event):
    print("回调函数.")
    try:
        print("尝试读取动作文件")
        with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myaction.pkl','rb') as f:
            Myaction = pickle.load(f) 
        if Myaction['action'] is not None:
            action = Myaction['action']
            over_flag=Myaction['over_flag']
            Myaction['action']= None                #清空文件 避免一次文件被多次使用
            with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myaction.pkl', 'wb') as f:
                pickle.dump(Myaction, f)
            do_my_action(action,over_flag)    #使用文件里的内容去更新一次   
        else:
            print("Myaction None") 
    except Exception as e:
        time.sleep(0.001)
        pass

# ===== 定时器（核心控制逻辑） =====
def do_my_action(action,over_flag):
    global abs_y_state, abs_rx_state, abs_ry_state
    global flag_curve, flag_forward, tail_point, camera, Tree_Point
    global smooth_rx, smooth_ry
    print("执行动作.")
    update_flag = -1 #更新标志位


    if over_flag == 1:
        tail_point.position = [0, 0, 0]
        tail_point.direction = [1, 0, 0]
        smooth_rx = 0.0
        smooth_ry = 0.0
        print("超出长度，重新归零！")

    # -------------------------------
    # 1) 读取手柄输入（原始）
    # -------------------------------
    raw_y  = action[0]
    if raw_y>0:
        raw_y=1
    else:
        raw_y=0.5
    raw_rx = action[1]
    raw_ry = action[2]

    print(f"Raw Input  RX={raw_rx}, RY={raw_ry}, Y={raw_y}")

    abs_y_state  = round(raw_y, 3)
    abs_rx_state = round(raw_rx, 3)
    abs_ry_state = round(raw_ry, 3)

    # -------------------------------
    # 2) 对 XY 输入做平滑（核心修改）
    # -------------------------------
    alpha = 0.1
    max_step = 0.03   # 摆动上限：每次回调 smooth 最多变化 0.03（按你手柄归一化范围调）

    dx = raw_rx - smooth_rx
    dy = raw_ry - smooth_ry

    step_x = np.sign(dx) * min(abs(alpha * dx), max_step)
    step_y = np.sign(dy) * min(abs(alpha * dy), max_step)

    candidate_rx = smooth_rx + step_x
    candidate_ry = smooth_ry + step_y

    #print(f"Candidate Smooth RX={candidate_rx}, RY={candidate_ry}")

    if is_valid_bending(candidate_rx, candidate_ry):
        smooth_rx = candidate_rx
        smooth_ry = candidate_ry
    else:
        # 越界 → 不更新（停在最大合法值）
        pass    

    #print(f"Smooth Input RX={smooth_rx}, RY={smooth_ry}")
    # -------------------------------
    # 3) 平滑后的 XY → 映射到角度
    # -------------------------------
    Theta = mf.Solve_theta(smooth_rx, smooth_ry)
    phi   = 2.5 * mf.Flag_curve(smooth_rx, smooth_ry)

    # 选取机器人模型形状（也用平滑后的手柄 XY）
    flag_curve = mf.Flag_curve(smooth_rx, smooth_ry)

    #print(f"[Smooth θ, φ]  Theta={Theta:.2f}, Phi={phi:.2f}")

    # -------------------------------
    # 4) 前进方向
    # -------------------------------
    # 获取头部点信息
    HandPosition, HandDirection = tail_point.Hand_Point(Theta, phi)
    HandPosition = [round(c, 3) for c in HandPosition]

    # 最近点查找（基于当前状态）
    TailNear = mf.find_closest_point(np.array(tail_point.position), Tree_Point)
    HandNear = mf.find_closest_point(np.array(HandPosition), Tree_Point)

    # 前进信号：只允许 0/1（你要求 forward_raw > 0 才前进）
    forward_raw = mf.Flag_forward(raw_y)
    #forward_raw = 1 if forward_raw > 0 else 0
    k = 0.6* raw_y

    #print(f"oldhandposition: {HandPosition}")

    # ================================
    # 1) 保存旧状态（用于回滚）
    # ================================
    old_pos = np.array(tail_point.position, dtype=np.float64).copy() #尾部点位置和姿态
    old_dir = np.array(tail_point.direction, dtype=np.float64).copy()
    TransOld = tail_point.Translate_M(Theta)

    # ================================
    # 2) 按正常逻辑计算候选 new_pos / new_dir
    # ================================
    M_vector = mf.normalize_vector(HandNear - TailNear)

    new_pos = old_pos + (1 * k) * np.array(HandDirection, dtype=np.float64)

    # 球面限制（中心线半径约束）
    R_limit = 5.0  # 原注释里就是 5；需要改就改这个
    tlp_near = mf.find_closest_point(np.array(new_pos), mylinepointsall)
    vec = new_pos - tlp_near
    dist = np.linalg.norm(vec)
    if dist > R_limit:
        new_pos = tlp_near + R_limit * vec / (dist + 1e-12)

    new_dir = old_dir + 0.05 * k * np.array(M_vector, dtype=np.float64)
    new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-12)

    # ================================
    # 3) 试探应用候选状态，算新的头部点是否“出界”
    # ================================
    tail_point.position = new_pos
    tail_point.direction = new_dir

    TransNew = tail_point.Translate_M(Theta)
    #robot_actor.SetMapper(Pmappers[flag_curve])
    #robot_actor.SetUserTransform(TransNew)

    # 用候选状态重新算头部点
    HandPosition_new, HandDirection_new = tail_point.Hand_Point(Theta, phi)
    HandPosition_new = [round(c, 3) for c in HandPosition_new]

    # 你定义的卡住：头部点在气道模型外部
    head_stuck = is_outside_airway(HandPosition_new)   # True => 内部 => 回滚

    # ================================
    # 4) 提交/回滚
    # ================================
    #print(f"head_stuck: {head_stuck}")
    if not head_stuck: #如果在内部
        # 回滚：不采用这次前进（保持旧位置/方向/姿态）
        tail_point.position = old_pos
        tail_point.direction = old_dir

        #robot_actor.SetMapper(Pmappers[flag_curve])
        robot_actor.SetUserTransform(TransOld)
        update_flag == 1
        print("Head stuck (outside airway) -> rollback.")
    else:
        # 提交：已经是新状态，无需再做事
        robot_actor.SetUserTransform(TransNew)
        pass

    # （渲染放在你 timer_callback 末尾统一 Render 即可）

    # 中心线距离更新后得到头部点位置
    HandPosition, HandDirection = tail_point.Hand_Point(Theta, phi)
    HandPosition = [round(c, 3) for c in HandPosition]

    HandNearPointcl=mf.find_closest_point(np.array(HandPosition, dtype=np.float32),mylinepoints1) #离头部点最近的目标中心线的点
    Handdistancecl=np.linalg.norm(np.array(HandPosition) - np.array(HandNearPointcl)) #计算距离

    #计算完成率
    idx, dist = mf.find_nearest_point(HandPosition, mylinepoints1)
    completion_ratio = mf.calculate_completion(idx, linepointnum)
    completion_rate=completion_ratio[0][0]
    #print(f"Completion Ratio: {completion_rate*100}%")

    #判断更新后状态
    if  Handdistancecl >= 10:#头部点离轴线距离大于8mm
    #if  Handdistance >= 50:
        print("超出空间！")
        update_flag = 1
        M_vector=np.array([0,0,0])

    #计算现在点到目标点向量
    positiontotarget=target_to_current(target_point,HandPosition)
    distiance_postiontotarget=np.linalg.norm(positiontotarget)

    threshold=10  #判断距离目标点阈值

    #判断是否到达目标点 阈值threshold
    if distiance_postiontotarget <= threshold:
        update_flag = 2
    else:
        pass
    #用done_flag来判断是否结束
    if update_flag == 1:
        tail_point.position = [0, 0, 0]
        tail_point.direction = [1, 0, 0]
        smooth_rx = 0.0
        smooth_ry = 0.0
        print("重新开始，偏差过大！")
    if update_flag == 2:
        tail_point.position = [0, 0, 0]
        tail_point.direction = [1, 0, 0]
        smooth_rx = 0.0
        smooth_ry = 0.0
        print("重新开始,达到目标！")


    # 调整相机
    HPos, HDir = HandPosition, HandDirection
    camera.SetPosition(HPos)
    camera.SetFocalPoint(HPos[0] + HDir[0], HPos[1] + HDir[1], HPos[2] + HDir[2])
    camera.SetViewUp(0, 1, 0)
    camera.SetClippingRange(0.001, 1000)

    cam_pos = camera.GetPosition()
    cam_fp = camera.GetFocalPoint()
    scope_light.SetPosition(cam_pos)
    scope_light.SetFocalPoint(cam_fp)

    ren_win.Render()
    time.sleep(0.001)

    try:
        current_image = capture_image(ren_win)
        #print(current_image.shape)

        #print(resized_image.shape)
    except Exception as e:
        print("error")

    Obv_11=[positiontotarget[0],positiontotarget[1],positiontotarget[2],
            HPos[0],HPos[1],HPos[2],
            HDir[0],HDir[1],HDir[2],
            Handdistancecl,completion_rate,
            distiance_postiontotarget,
            abs_y_state,abs_rx_state,abs_ry_state,
            ]

    read_flag=1

    observation_dict = {
        "image": current_image,
        "Sensor11": Obv_11, #计算距离终点,观察空间，向量
        "deviation_position": Handdistancecl, #计算头部点偏离中心线距离 计算奖励
        "update_flag": update_flag,
        "read_flag":read_flag
    }   #print(observation_dict)
    
    lock = FileLock("D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myobservation.pkl.lock")
    with lock:
        with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myobservation.pkl', 'wb') as f:
            pickle.dump(observation_dict, f)

# ===== 入口 =====
if __name__ == "__main__":
    main()

#D:\Code\RL_BiSST\RL_Bisst\data_exchange