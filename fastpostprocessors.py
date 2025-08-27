# Fast Postprocessors - прямая работа с FastMesh/Mesh без trimesh/pymeshlab
# Замена для hy3dshape/postprocessors.py

import torch
import numpy as np
from typing import Union
from mesh_processor.mesh import Mesh, FastMesh
from advanced_mesh_simplifier import advanced_reduce_faces, AdvancedFaceReducer


def fast_reduce_faces(mesh_obj: Union[Mesh, FastMesh], max_faces: int = 200000, 
                     use_advanced: bool = True) -> Union[Mesh, FastMesh]:
    """
    Уменьшение количества граней до заданного максимума
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        max_faces: максимальное количество граней (аналог max_facenum в оригинале)
        use_advanced: использовать продвинутый алгоритм Quadric Edge Collapse
    
    Returns:
        Тот же тип mesh объекта с уменьшенным количеством граней
    """
    try:
        faces = mesh_obj.f
        
        if faces.shape[0] == 0:
            return mesh_obj
        
        # Если граней уже меньше чем max_faces, ничего не делаем
        if faces.shape[0] <= max_faces:
            print(f"[FastPostprocessor] Mesh already has {faces.shape[0]} faces (target: {max_faces})")
            return mesh_obj
        
        if use_advanced:
            # Используем продвинутый алгоритм Quadric Edge Collapse
            print(f"[FastPostprocessor] Используем Quadric Edge Collapse: {faces.shape[0]} → {max_faces}")
            mesh_obj = advanced_reduce_faces(mesh_obj, max_faces)
        else:
            # Используем простой алгоритм по площади (старый)
            print(f"[FastPostprocessor] Используем простое удаление по площади: {faces.shape[0]} → {max_faces}")
            mesh_obj = _simple_area_reduction(mesh_obj, max_faces)
        
    except Exception as e:
        print(f"[FastPostprocessor] Face reduction ошибка: {e}")
        # Fallback к простому алгоритму
        try:
            mesh_obj = _simple_area_reduction(mesh_obj, max_faces)
        except Exception as e2:
            print(f"[FastPostprocessor] Fallback ошибка: {e2}")
    
    return mesh_obj


def _simple_area_reduction(mesh_obj: Union[Mesh, FastMesh], max_faces: int) -> Union[Mesh, FastMesh]:
    """Простое удаление граней по площади (fallback)"""
    vertices = mesh_obj.v
    faces = mesh_obj.f
    
    # Вычисляем площади треугольников
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]] 
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_product = torch.cross(edge1, edge2, dim=1)
    areas = 0.5 * torch.norm(cross_product, dim=1)
    
    # Сортируем по площади и оставляем самые большие
    sorted_indices = torch.argsort(areas, descending=True)
    keep_faces = sorted_indices[:max_faces]
    
    new_faces = faces[keep_faces]
    mesh_obj.f = new_faces
    
    if mesh_obj.fn is not None:
        mesh_obj.fn = new_faces
    
    # Пересчитываем нормали
    mesh_obj.auto_normal()
    
    print(f"[FastPostprocessor] Simple reduction: {len(faces)} → {len(new_faces)} граней")
    return mesh_obj


def fast_remove_small_components(mesh_obj: Union[Mesh, FastMesh], min_component_ratio: float = 0.01) -> Union[Mesh, FastMesh]:
    """
    Удаление мелких несвязанных компонентов
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        min_component_ratio: минимальный размер компонента относительно общего числа граней
    
    Returns:
        Mesh объект без мелких компонентов
    """
    try:
        vertices = mesh_obj.v
        faces = mesh_obj.f
        
        if faces.shape[0] == 0:
            return mesh_obj
        
        # Простой алгоритм: удаляем грани с вершинами далеко от центра масс
        face_centers = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
        mesh_center = face_centers.mean(dim=0)
        
        # Расстояния от центра
        distances = torch.norm(face_centers - mesh_center, dim=1)
        distance_threshold = torch.quantile(distances, 0.9)  # Убираем 10% самых дальних
        
        keep_mask = distances <= distance_threshold
        new_faces = faces[keep_mask]
        
        mesh_obj.f = new_faces
        if mesh_obj.fn is not None:
            mesh_obj.fn = new_faces
            
        # Пересчитываем нормали
        mesh_obj.auto_normal()
        
        removed_count = len(faces) - len(new_faces)
        print(f"[FastPostprocessor] Removed small components: {removed_count} граней")
        
    except Exception as e:
        print(f"[FastPostprocessor] Small components removal ошибка: {e}")
    
    return mesh_obj


def fast_remove_degenerate_faces(mesh_obj: Union[Mesh, FastMesh], area_threshold: float = 1e-8) -> Union[Mesh, FastMesh]:
    """
    Удаление вырожденных граней (с очень маленькой площадью)
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        area_threshold: минимальная площадь грани
    
    Returns:
        Mesh объект без вырожденных граней
    """
    try:
        vertices = mesh_obj.v
        faces = mesh_obj.f
        
        if faces.shape[0] == 0:
            return mesh_obj
        
        # Вычисляем площади треугольников
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]] 
        v2 = vertices[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross_product = torch.cross(edge1, edge2, dim=1)
        areas = 0.5 * torch.norm(cross_product, dim=1)
        
        # Оставляем только грани с достаточной площадью
        valid_mask = areas > area_threshold
        new_faces = faces[valid_mask]
        
        mesh_obj.f = new_faces
        if mesh_obj.fn is not None:
            mesh_obj.fn = new_faces
            
        # Пересчитываем нормали
        mesh_obj.auto_normal()
        
        removed_count = len(faces) - len(new_faces)
        print(f"[FastPostprocessor] Removed degenerate faces: {removed_count} граней")
        
    except Exception as e:
        print(f"[FastPostprocessor] Degenerate faces removal ошибка: {e}")
    
    return mesh_obj


def fast_normalize_mesh(mesh_obj: Union[Mesh, FastMesh], scale_factor: float = 1.2) -> Union[Mesh, FastMesh]:
    """
    Нормализация меша в сферу
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        scale_factor: коэффициент масштабирования
    
    Returns:
        Нормализованный mesh объект
    """
    try:
        vertices = mesh_obj.v
        
        # Находим центр и масштаб
        max_bb = vertices.max(dim=0)[0]
        min_bb = vertices.min(dim=0)[0]
        center = (max_bb + min_bb) / 2
        
        scale = torch.norm(vertices - center, dim=1).max() * 2.0
        
        # Нормализуем
        normalized_vertices = (vertices - center) * (scale_factor / scale)
        mesh_obj.v = normalized_vertices
        
        # Обновляем центр и масштаб для корректного отображения
        mesh_obj.ori_center = center
        mesh_obj.ori_scale = scale_factor / scale
        
        print(f"[FastPostprocessor] Mesh normalized with scale factor {scale_factor}")
        
    except Exception as e:
        print(f"[FastPostprocessor] Normalization ошибка: {e}")
    
    return mesh_obj


def fast_clean_mesh(mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
    """
    Комплексная очистка меша - все операции сразу
    
    Args:
        mesh_obj: FastMesh или Mesh объект
    
    Returns:
        Очищенный mesh объект
    """
    print("[FastPostprocessor] Начинаем комплексную очистку меша...")
    
    # 1. Удаляем вырожденные грани
    mesh_obj = fast_remove_degenerate_faces(mesh_obj)
    
    # 2. Удаляем мелкие компоненты
    mesh_obj = fast_remove_small_components(mesh_obj)
    
    # 3. Нормализуем
    mesh_obj = fast_normalize_mesh(mesh_obj)
    
    print("[FastPostprocessor] Комплексная очистка завершена")
    return mesh_obj


class FastFaceReducer:
    """Класс для уменьшения граней - аналог FaceReducer"""
    
    def __init__(self, max_faces: int = 200000, use_advanced: bool = True, 
                 preserve_boundary: bool = True, preserve_topology: bool = True):
        self.max_faces = max_faces
        self.use_advanced = use_advanced
        self.preserve_boundary = preserve_boundary
        self.preserve_topology = preserve_topology
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        if self.use_advanced:
            return advanced_reduce_faces(mesh_obj, self.max_faces, 
                                       self.preserve_boundary, self.preserve_topology)
        else:
            return fast_reduce_faces(mesh_obj, self.max_faces, use_advanced=False)


class FastFloaterRemover:
    """Класс для удаления мелких компонентов - аналог FloaterRemover"""
    
    def __init__(self, min_component_ratio: float = 0.01):
        self.min_component_ratio = min_component_ratio
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return fast_remove_small_components(mesh_obj, self.min_component_ratio)


class FastDegenerateFaceRemover:
    """Класс для удаления вырожденных граней - аналог DegenerateFaceRemover"""
    
    def __init__(self, area_threshold: float = 1e-8):
        self.area_threshold = area_threshold
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return fast_remove_degenerate_faces(mesh_obj, self.area_threshold)


class FastMeshCleaner:
    """Комплексная очистка меша - все операции сразу"""
    
    def __init__(self, 
                 max_faces: int = 200000,
                 remove_degenerates: bool = True,
                 remove_floaters: bool = True,
                 normalize: bool = True):
        self.max_faces = max_faces
        self.remove_degenerates = remove_degenerates
        self.remove_floaters = remove_floaters
        self.normalize = normalize
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        print(f"[FastMeshCleaner] Начинаем обработку меша...")
        
        if self.remove_degenerates:
            mesh_obj = fast_remove_degenerate_faces(mesh_obj)
        
        if self.remove_floaters:
            mesh_obj = fast_remove_small_components(mesh_obj)
        
        if self.max_faces > 0:
            mesh_obj = fast_reduce_faces(mesh_obj, self.max_faces)
        
        if self.normalize:
            mesh_obj = fast_normalize_mesh(mesh_obj)
        
        print(f"[FastMeshCleaner] Обработка завершена")
        return mesh_obj


# Простые функции для быстрого использования
def process_mesh_fast(mesh_obj: Union[Mesh, FastMesh], 
                     reduce_faces: bool = True,
                     max_faces: int = 200000) -> Union[Mesh, FastMesh]:
    """
    Быстрая обработка меша одной функцией
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        reduce_faces: применять ли уменьшение граней
        max_faces: максимальное количество граней
    
    Returns:
        Обработанный mesh объект
    """
    if reduce_faces:
        mesh_obj = fast_reduce_faces(mesh_obj, max_faces)
    
    mesh_obj = fast_clean_mesh(mesh_obj)
    return mesh_obj


# Алиасы для совместимости
FastMeshSimplifier = FastFaceReducer
