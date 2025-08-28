# PyMeshLab Wrapper - используем проверенные алгоритмы pymeshlab с нашими FastMesh/Mesh
# Конвертируем наш формат → pymeshlab → обрабатываем → конвертируем обратно

import tempfile
import os
import torch
import numpy as np
import pymeshlab
from typing import Union
from mesh_processor.mesh import Mesh, FastMesh


def to_pymeshlab(mesh_obj: Union[Mesh, FastMesh]) -> pymeshlab.MeshSet:
    """
    Конвертирует наш FastMesh/Mesh в pymeshlab.MeshSet
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        
    Returns:
        pymeshlab.MeshSet объект
    """
    
    # Извлекаем данные
    vertices = mesh_obj.v.detach().cpu().numpy().astype(np.float64)
    faces = mesh_obj.f.detach().cpu().numpy().astype(np.uint32)
    
    # Создаем MeshSet
    ms = pymeshlab.MeshSet()
    
    # Создаем Mesh из вершин и граней
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(mesh, "converted_mesh")
    
    return ms


def from_pymeshlab(ms: pymeshlab.MeshSet, original_mesh: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
    """
    Конвертирует pymeshlab.MeshSet обратно в наш FastMesh/Mesh
    
    Args:
        ms: pymeshlab.MeshSet объект
        original_mesh: оригинальный mesh для сохранения типа и device
        
    Returns:
        FastMesh или Mesh объект (тот же тип что был)
    """
    
    # Извлекаем данные из pymeshlab
    current_mesh = ms.current_mesh()
    vertices = current_mesh.vertex_matrix().astype(np.float32)
    faces = current_mesh.face_matrix().astype(np.int32)
    
    # Определяем device и тип
    device = original_mesh.v.device
    mesh_type = type(original_mesh)
    
    # Создаем новый объект того же типа
    new_mesh = mesh_type(device=device)
    new_mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
    new_mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
    
    # Копируем дополнительные атрибуты если они были
    if hasattr(original_mesh, 'albedo') and original_mesh.albedo is not None:
        new_mesh.albedo = original_mesh.albedo
    if hasattr(original_mesh, 'metallicRoughness') and original_mesh.metallicRoughness is not None:
        new_mesh.metallicRoughness = original_mesh.metallicRoughness
    if hasattr(original_mesh, 'vt') and original_mesh.vt is not None:
        new_mesh.vt = original_mesh.vt
    if hasattr(original_mesh, 'ft') and original_mesh.ft is not None:
        new_mesh.ft = original_mesh.ft
    
    # Пересчитываем нормали
    new_mesh.auto_normal()
    
    return new_mesh


def pymeshlab_reduce_faces(mesh_obj: Union[Mesh, FastMesh], max_faces: int = 40000,
                          quality_threshold: float = 1.0, preserve_boundary: bool = True,
                          boundary_weight: int = 3, preserve_normal: bool = True,
                          preserve_topology: bool = True, autoclean: bool = True) -> Union[Mesh, FastMesh]:
    """
    Уменьшение граней используя оригинальный pymeshlab алгоритм
    Точная копия reduce_face() из оригинального postprocessors.py
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        max_faces: максимальное количество граней
        quality_threshold: порог качества
        preserve_boundary: сохранять границы
        boundary_weight: вес границ
        preserve_normal: сохранять нормали
        preserve_topology: сохранять топологию
        autoclean: автоочистка
        
    Returns:
        Упрощенный mesh объект того же типа
    """
    
    try:
        # Если граней уже меньше, ничего не делаем
        if mesh_obj.f.shape[0] <= max_faces:
            print(f"[PyMeshLabWrapper] Mesh already has {mesh_obj.f.shape[0]} faces (target: {max_faces})")
            return mesh_obj
        
        print(f"[PyMeshLabWrapper] Reducing faces: {mesh_obj.f.shape[0]} → {max_faces}")
        
        # Конвертируем в pymeshlab
        ms = to_pymeshlab(mesh_obj)
        
        # Применяем фильтр (точно как в оригинале)
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=max_faces,
            qualitythr=quality_threshold,
            preserveboundary=preserve_boundary,
            boundaryweight=boundary_weight,
            preservenormal=preserve_normal,
            preservetopology=preserve_topology,
            autoclean=autoclean
        )
        
        # Конвертируем обратно
        result_mesh = from_pymeshlab(ms, mesh_obj)
        
        print(f"[PyMeshLabWrapper] Face reduction completed: {result_mesh.f.shape[0]} faces")
        return result_mesh
        
    except Exception as e:
        print(f"[PyMeshLabWrapper] Face reduction error: {e}")
        return mesh_obj


def pymeshlab_remove_floaters(mesh_obj: Union[Mesh, FastMesh], 
                             face_ratio: float = 0.005) -> Union[Mesh, FastMesh]:
    """
    Удаление мелких компонентов используя pymeshlab
    Точная копия remove_floater() из оригинального postprocessors.py
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        face_ratio: минимальный размер компонента (отношение к общему числу граней)
        
    Returns:
        Mesh объект без мелких компонентов
    """
    
    try:
        print(f"[PyMeshLabWrapper] Removing floaters with ratio {face_ratio}")
        
        # Конвертируем в pymeshlab
        ms = to_pymeshlab(mesh_obj)
        
        # Применяем фильтры (точно как в оригинале)
        ms.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                        nbfaceratio=face_ratio)
        ms.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
        ms.apply_filter("meshing_remove_selected_vertices_and_faces")
        
        # Конвертируем обратно
        result_mesh = from_pymeshlab(ms, mesh_obj)
        
        print(f"[PyMeshLabWrapper] Floater removal completed")
        return result_mesh
        
    except Exception as e:
        print(f"[PyMeshLabWrapper] Floater removal error: {e}")
        return mesh_obj


def pymeshlab_remove_degenerates(mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
    """
    Удаление вырожденных граней используя pymeshlab
    Аналог DegenerateFaceRemover из оригинального postprocessors.py
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        
    Returns:
        Mesh объект без вырожденных граней
    """
    
    try:
        print(f"[PyMeshLabWrapper] Removing degenerate faces")
        
        # Конвертируем в pymeshlab
        ms = to_pymeshlab(mesh_obj)
        
        # Применяем очистку через сохранение/загрузку (как в оригинале)
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
            ms.save_current_mesh(temp_file.name)
            ms_new = pymeshlab.MeshSet()
            ms_new.load_new_mesh(temp_file.name)
            os.unlink(temp_file.name)  # Удаляем временный файл
        
        # Конвертируем обратно
        result_mesh = from_pymeshlab(ms_new, mesh_obj)
        
        print(f"[PyMeshLabWrapper] Degenerate face removal completed")
        return result_mesh
        
    except Exception as e:
        print(f"[PyMeshLabWrapper] Degenerate face removal error: {e}")
        return mesh_obj


def pymeshlab_clean_mesh(mesh_obj: Union[Mesh, FastMesh], 
                        max_faces: int = 40000,
                        remove_floaters: bool = True,
                        remove_degenerates: bool = True) -> Union[Mesh, FastMesh]:
    """
    Комплексная очистка меша используя pymeshlab
    Комбинирует все операции из оригинального postprocessors.py
    
    Args:
        mesh_obj: FastMesh или Mesh объект
        max_faces: максимальное количество граней
        remove_floaters: удалять мелкие компоненты
        remove_degenerates: удалять вырожденные грани
        
    Returns:
        Очищенный mesh объект
    """
    
    print(f"[PyMeshLabWrapper] Starting comprehensive mesh cleaning")
    
    result_mesh = mesh_obj
    
    # 1. Удаляем вырожденные грани
    if remove_degenerates:
        result_mesh = pymeshlab_remove_degenerates(result_mesh)
    
    # 2. Удаляем мелкие компоненты
    if remove_floaters:
        result_mesh = pymeshlab_remove_floaters(result_mesh)
    
    # 3. Уменьшаем количество граней
    if max_faces > 0:
        result_mesh = pymeshlab_reduce_faces(result_mesh, max_faces)
    
    print(f"[PyMeshLabWrapper] Comprehensive mesh cleaning completed")
    return result_mesh


# Классы-обертки для совместимости

class PyMeshLabFaceReducer:
    """Класс для уменьшения граней через pymeshlab - аналог FaceReducer"""
    
    def __init__(self, max_faces: int = 40000):
        self.max_faces = max_faces
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return pymeshlab_reduce_faces(mesh_obj, self.max_faces)


class PyMeshLabFloaterRemover:
    """Класс для удаления мелких компонентов через pymeshlab - аналог FloaterRemover"""
    
    def __init__(self, face_ratio: float = 0.005):
        self.face_ratio = face_ratio
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return pymeshlab_remove_floaters(mesh_obj, self.face_ratio)


class PyMeshLabDegenerateFaceRemover:
    """Класс для удаления вырожденных граней через pymeshlab - аналог DegenerateFaceRemover"""
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return pymeshlab_remove_degenerates(mesh_obj)


class PyMeshLabMeshCleaner:
    """Комплексная очистка меша через pymeshlab"""
    
    def __init__(self, max_faces: int = 40000, remove_floaters: bool = True, 
                 remove_degenerates: bool = True):
        self.max_faces = max_faces
        self.remove_floaters = remove_floaters
        self.remove_degenerates = remove_degenerates
    
    def __call__(self, mesh_obj: Union[Mesh, FastMesh]) -> Union[Mesh, FastMesh]:
        return pymeshlab_clean_mesh(mesh_obj, self.max_faces, 
                                  self.remove_floaters, self.remove_degenerates)
