"""
FastGLB - быстрая библиотека для работы с GLB/GLTF файлами
Замена trimesh для GLB/GLTF без зависимостей
"""

import os
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .io_gltf import load_gltf_or_glb, get_binary_data
from .mesh_ops import get_all_meshes_triangles
from .accessors import access_data


class FastGLB:
    """Основной класс для работы с GLB/GLTF файлами"""
    
    def __init__(self, vertices=None, faces=None, vertex_normals=None, 
                 vertex_colors=None, uv_coordinates=None, textures=None):
        self.vertices = vertices if vertices is not None else np.array([]).reshape(0, 3)
        self.faces = faces if faces is not None else np.array([]).reshape(0, 3)
        self.vertex_normals = vertex_normals
        self.vertex_colors = vertex_colors
        self.uv_coordinates = uv_coordinates
        self.textures = textures or {}
        self._visual = None
    
    @property
    def visual(self):
        """Визуальные свойства"""
        if self._visual is None:
            self._visual = FastGLBVisual(self)
        return self._visual
    
    def export(self, path):
        """Экспорт в файл (пока что заглушка)"""
        # TODO: Реализовать экспорт через наш mesh processor
        raise NotImplementedError("Export будет реализован позже")
    
    def apply_transform(self, matrix):
        """Применение трансформации"""
        if len(self.vertices) == 0:
            return self
        
        # Применяем трансформацию к вершинам
        vertices_h = np.column_stack([self.vertices, np.ones(len(self.vertices))])
        transformed_vertices = (matrix @ vertices_h.T).T[:, :3]
        
        # Применяем трансформацию к нормалям (только поворот)
        transformed_normals = None
        if self.vertex_normals is not None:
            rotation_matrix = matrix[:3, :3]
            transformed_normals = (rotation_matrix @ self.vertex_normals.T).T
        
        return FastGLB(
            vertices=transformed_vertices,
            faces=self.faces.copy() if len(self.faces) > 0 else None,
            vertex_normals=transformed_normals,
            vertex_colors=self.vertex_colors.copy() if self.vertex_colors is not None else None,
            uv_coordinates=self.uv_coordinates.copy() if self.uv_coordinates is not None else None,
            textures=self.textures.copy()
        )
    
    def copy(self):
        """Создание копии"""
        return FastGLB(
            vertices=self.vertices.copy() if len(self.vertices) > 0 else None,
            faces=self.faces.copy() if len(self.faces) > 0 else None,
            vertex_normals=self.vertex_normals.copy() if self.vertex_normals is not None else None,
            vertex_colors=self.vertex_colors.copy() if self.vertex_colors is not None else None,
            uv_coordinates=self.uv_coordinates.copy() if self.uv_coordinates is not None else None,
            textures=self.textures.copy()
        )


class FastGLBVisual:
    """Визуальные свойства FastGLB"""
    
    def __init__(self, fastglb_obj):
        self.fastglb = fastglb_obj
        self._material = None
    
    @property
    def kind(self):
        """Тип визуализации"""
        if self.fastglb.vertex_colors is not None:
            return 'vertex'
        elif self.fastglb.textures or self.fastglb.uv_coordinates is not None:
            return 'texture'
        else:
            return 'vertex'
    
    @property
    def vertex_colors(self):
        """Цвета вершин в формате trimesh (RGBA, 0-255)"""
        if self.fastglb.vertex_colors is not None:
            colors = self.fastglb.vertex_colors
            if colors.shape[1] == 3:
                # Добавляем альфа канал
                alpha = np.ones((colors.shape[0], 1))
                colors = np.concatenate([colors, alpha], axis=1)
            return (colors * 255).astype(np.uint8)
        else:
            # Белый цвет по умолчанию
            num_vertices = len(self.fastglb.vertices)
            return np.full((num_vertices, 4), 255, dtype=np.uint8)
    
    @property
    def uv(self):
        """UV координаты"""
        if self.fastglb.uv_coordinates is not None:
            # В GLB V координата может быть перевернута
            uv = self.fastglb.uv_coordinates.copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # Переворачиваем как в trimesh
            return uv
        return None
    
    @property
    def material(self):
        """Материал"""
        if self._material is None:
            self._material = FastGLBMaterial(self.fastglb)
        return self._material


class FastGLBMaterial:
    """Материал FastGLB"""
    
    def __init__(self, fastglb_obj):
        self.fastglb = fastglb_obj
    
    @property
    def baseColorTexture(self):
        """Основная текстура"""
        return self.fastglb.textures.get('albedo')
    
    @property
    def metallicRoughnessTexture(self):
        """Metallic-Roughness текстура"""
        return self.fastglb.textures.get('metallicRoughness')
    
    def to_pbr(self):
        """Конвертация в PBR (возвращает self)"""
        return self


class FastGLBScene:
    """Scene объект для совместимости"""
    
    def __init__(self, geometry_dict):
        self.geometry = geometry_dict
        self.graph = None  # Заглушка
    
    def dump(self, concatenate=True):
        """Объединение всех мешей в сцене"""
        if not self.geometry:
            return FastGLB()
        
        if len(self.geometry) == 1:
            return list(self.geometry.values())[0]
        
        if concatenate:
            return concatenate_fastglb_meshes(list(self.geometry.values()))
        else:
            return list(self.geometry.values())


def load(path: str) -> FastGLB:
    """Главная функция загрузки - замена trimesh.load()"""
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext in ['.glb', '.gltf']:
        return load_glb(path)
    elif file_ext == '.obj':
        return load_obj(path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def load_glb(path: str) -> FastGLB:
    """Загрузка GLB/GLTF файла"""
    
    try:
        # Загружаем GLTF документ
        doc = load_gltf_or_glb(path)
        print(f"[FastGLB] Загружен GLTF документ: {len(doc.meshes())} мешей")
        
        # Извлекаем геометрию
        vertices, faces, mesh_groups = get_all_meshes_triangles(doc, transform_to_global=True)
        
        if len(vertices) == 0:
            print("[FastGLB] Нет геометрии в файле")
            return FastGLB()
        
        print(f"[FastGLB] Извлечено: {len(vertices)} вершин, {len(faces)} граней")
        
        # Извлекаем дополнительные данные из первого меша
        vertex_normals = None
        vertex_colors = None
        uv_coordinates = None
        textures = {}
        
        if doc.meshes():
            first_mesh = doc.meshes()[0]
            for primitive in first_mesh.get("primitives", []):
                attributes = primitive.get("attributes", {})
                
                # UV координаты
                if "TEXCOORD_0" in attributes:
                    try:
                        uv_accessor_idx = int(attributes["TEXCOORD_0"])
                        uv_coordinates = access_data(doc, uv_accessor_idx).astype(np.float32)
                        print(f"[FastGLB] UV координаты: {uv_coordinates.shape}")
                    except Exception as e:
                        print(f"[FastGLB] Ошибка UV: {e}")
                
                # Нормали
                if "NORMAL" in attributes:
                    try:
                        normal_accessor_idx = int(attributes["NORMAL"])
                        vertex_normals = access_data(doc, normal_accessor_idx).astype(np.float32)
                        print(f"[FastGLB] Нормали: {vertex_normals.shape}")
                    except Exception as e:
                        print(f"[FastGLB] Ошибка нормалей: {e}")
                
                # Цвета вершин
                if "COLOR_0" in attributes:
                    try:
                        color_accessor_idx = int(attributes["COLOR_0"])
                        vertex_colors = access_data(doc, color_accessor_idx).astype(np.float32)
                        if vertex_colors.shape[1] > 3:
                            vertex_colors = vertex_colors[:, :3]
                        print(f"[FastGLB] Цвета вершин: {vertex_colors.shape}")
                    except Exception as e:
                        print(f"[FastGLB] Ошибка цветов: {e}")
        
        # Извлекаем текстуры
        textures = extract_textures_from_gltf(doc)
        
        return FastGLB(
            vertices=vertices,
            faces=faces,
            vertex_normals=vertex_normals,
            vertex_colors=vertex_colors,
            uv_coordinates=uv_coordinates,
            textures=textures
        )
        
    except Exception as e:
        print(f"[FastGLB] Ошибка загрузки {path}: {e}")
        raise e


def load_obj(path: str) -> FastGLB:
    """Загрузка OBJ файла (простая реализация)"""
    # Пока что заглушка - можно реализовать позже
    raise NotImplementedError("OBJ загрузка через FastGLB пока не реализована")


def extract_textures_from_gltf(doc) -> Dict[str, np.ndarray]:
    """Извлекает текстуры из GLTF документа"""
    textures = {}
    
    try:
        if not (doc.materials() and doc.textures() and doc.images()):
            return textures
        
        # Берем первый материал
        material = doc.materials()[0]
        pbr = material.get("pbrMetallicRoughness", {})
        
        # Основная текстура (albedo)
        base_color_texture = pbr.get("baseColorTexture")
        if base_color_texture:
            texture_idx = base_color_texture.get("index", 0)
            albedo_texture = extract_texture_by_index(doc, texture_idx)
            if albedo_texture is not None:
                textures['albedo'] = albedo_texture
                print(f"[FastGLB] Извлечена albedo текстура: {albedo_texture.shape}")
        
        # Metallic-Roughness текстура
        metallic_roughness_texture = pbr.get("metallicRoughnessTexture")
        if metallic_roughness_texture:
            texture_idx = metallic_roughness_texture.get("index", 0)
            mr_texture = extract_texture_by_index(doc, texture_idx)
            if mr_texture is not None:
                textures['metallicRoughness'] = mr_texture
                print(f"[FastGLB] Извлечена metallic-roughness: {mr_texture.shape}")
        
    except Exception as e:
        print(f"[FastGLB] Ошибка извлечения текстур: {e}")
    
    return textures


def extract_texture_by_index(doc, texture_idx: int) -> Optional[np.ndarray]:
    """Извлекает текстуру по индексу"""
    try:
        if texture_idx >= len(doc.textures()):
            return None
        
        texture = doc.textures()[texture_idx]
        image_idx = texture.get("source", 0)
        
        if image_idx >= len(doc.images()):
            return None
        
        image_info = doc.images()[image_idx]
        
        # Извлекаем из буфера
        buffer_view_idx = image_info.get("bufferView")
        if buffer_view_idx is None:
            return None
        
        buffer_view = doc.bufferViews()[buffer_view_idx]
        buffer_idx = buffer_view.get("buffer", 0)
        byte_offset = buffer_view.get("byteOffset", 0)
        byte_length = buffer_view.get("byteLength", 0)
        
        binary_data = get_binary_data(doc, buffer_idx)
        image_bytes = binary_data[byte_offset:byte_offset + byte_length]
        
        # Декодируем изображение
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        texture_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if texture_image is not None:
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            return texture_image.astype(np.uint8)  # Оставляем в uint8 формате
        
    except Exception as e:
        print(f"[FastGLB] Ошибка извлечения текстуры {texture_idx}: {e}")
    
    return None


def concatenate_fastglb_meshes(meshes: List[FastGLB]) -> FastGLB:
    """Объединение нескольких FastGLB мешей"""
    if not meshes:
        return FastGLB()
    
    if len(meshes) == 1:
        return meshes[0]
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for mesh in meshes:
        if len(mesh.vertices) > 0:
            all_vertices.append(mesh.vertices)
            if len(mesh.faces) > 0:
                faces_with_offset = mesh.faces + vertex_offset
                all_faces.append(faces_with_offset)
            vertex_offset += len(mesh.vertices)
    
    if not all_vertices:
        return FastGLB()
    
    final_vertices = np.concatenate(all_vertices, axis=0)
    final_faces = np.concatenate(all_faces, axis=0) if all_faces else np.array([]).reshape(0, 3)
    
    # Берем текстуры и другие данные из первого меша
    first_mesh = meshes[0]
    
    return FastGLB(
        vertices=final_vertices,
        faces=final_faces,
        vertex_normals=first_mesh.vertex_normals,
        vertex_colors=first_mesh.vertex_colors,
        uv_coordinates=first_mesh.uv_coordinates,
        textures=first_mesh.textures
    )