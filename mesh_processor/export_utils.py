"""
Утилиты для экспорта из пайплайнов в наши mesh объекты
Прямые функции БЕЗ зависимости от trimesh
"""

import torch
import numpy as np
from .mesh import Mesh, FastMesh


def export_to_fastmesh(mesh_output, device=None):
    """
    Экспорт из пайплайна в FastMesh (замена export_to_trimesh_2_1)
    
    Args:
        mesh_output: объект с mesh_v и mesh_f от пайплайна
        device: torch device
    
    Returns:
        FastMesh объект или список FastMesh объектов
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                fastmesh = _convert_single_mesh_to_fastmesh(mesh, device)
                outputs.append(fastmesh)
        return outputs
    else:
        return _convert_single_mesh_to_fastmesh(mesh_output, device)


def export_to_mesh(mesh_output, device=None):
    """
    Экспорт из пайплайна в стандартный Mesh
    
    Args:
        mesh_output: объект с mesh_v и mesh_f от пайплайна
        device: torch device
    
    Returns:
        Mesh объект или список Mesh объектов
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                standard_mesh = _convert_single_mesh_to_mesh(mesh, device)
                outputs.append(standard_mesh)
        return outputs
    else:
        return _convert_single_mesh_to_mesh(mesh_output, device)


def _convert_single_mesh_to_fastmesh(mesh_obj, device):
    """Конвертирует один mesh объект в FastMesh"""
    
    try:
        # Извлекаем вершины и грани
        vertices = mesh_obj.mesh_v
        faces = mesh_obj.mesh_f
        
        # Переворачиваем грани (как в оригинальной функции)
        faces = faces[:, ::-1]
        
        # Конвертируем в numpy если нужно
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        print(f"[export_to_fastmesh] Конвертируем: {vertices.shape[0]} вершин, {faces.shape[0]} граней")
        
        # Создаем FastMesh
        fastmesh = FastMesh(device=device)
        fastmesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        fastmesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        
        # Автоматически генерируем нормали
        fastmesh.auto_normal()
        
        # Создаем пустую текстуру
        fastmesh._create_empty_albedo()
        
        print(f"[export_to_fastmesh] FastMesh создан успешно")
        return fastmesh
        
    except Exception as e:
        print(f"[export_to_fastmesh] Ошибка: {e}")
        return None


def _convert_single_mesh_to_mesh(mesh_obj, device):
    """Конвертирует один mesh объект в стандартный Mesh"""
    
    try:
        # Извлекаем вершины и грани
        vertices = mesh_obj.mesh_v
        faces = mesh_obj.mesh_f
        
        # Переворачиваем грани (как в оригинальной функции)
        faces = faces[:, ::-1]
        
        # Конвертируем в numpy если нужно
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        print(f"[export_to_mesh] Конвертируем: {vertices.shape[0]} вершин, {faces.shape[0]} граней")
        
        # Создаем стандартный Mesh
        mesh = Mesh(device=device)
        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        
        # Автоматически генерируем нормали
        mesh.auto_normal()
        
        # Создаем пустую текстуру
        mesh.set_new_albedo(1024, 1024)
        
        print(f"[export_to_mesh] Mesh создан успешно")
        return mesh
        
    except Exception as e:
        print(f"[export_to_mesh] Ошибка: {e}")
        return None


# Простые алиасы
export_to_fastmesh_2_1 = export_to_fastmesh
export_to_mesh_2_1 = export_to_mesh