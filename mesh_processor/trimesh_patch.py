"""
Патч для замены trimesh.load на FastGLB в существующем коде
"""

from . import fastglb
from .mesh import Mesh
import torch
import numpy as np


def patch_trimesh_load():
    """Заменяет trimesh.load на наш FastGLB для GLB файлов"""
    
    # Сохраняем оригинальный trimesh.load
    import trimesh
    original_trimesh_load = trimesh.load
    
    def patched_load(path, **kwargs):
        """Патченая версия trimesh.load"""
        
        if path.lower().endswith(('.glb', '.gltf')):
            print(f"🚀 [PATCH] Загрузка GLB через FastGLB: {path}")
            try:
                fastglb_mesh = fastglb.load(path)
                return FastGLBToTrimeshAdapter(fastglb_mesh)
            except Exception as e:
                print(f"❌ [PATCH] FastGLB ошибка, fallback к trimesh: {e}")
                return original_trimesh_load(path, **kwargs)
        else:
            # Для других форматов используем оригинальный trimesh
            return original_trimesh_load(path, **kwargs)
    
    # Заменяем trimesh.load
    trimesh.load = patched_load
    print("✅ trimesh.load запатчен для использования FastGLB")


class FastGLBToTrimeshAdapter:
    """Адаптер, который делает FastGLB объект похожим на trimesh"""
    
    def __init__(self, fastglb_mesh):
        self.fastglb_mesh = fastglb_mesh
        self._visual = None
    
    @property
    def vertices(self):
        return self.fastglb_mesh.vertices
    
    @property
    def faces(self):
        return self.fastglb_mesh.faces
    
    @property
    def vertex_normals(self):
        if self.fastglb_mesh.vertex_normals is not None:
            return self.fastglb_mesh.vertex_normals
        else:
            # Генерируем простые нормали
            return np.zeros_like(self.vertices)
    
    @property
    def visual(self):
        if self._visual is None:
            self._visual = FastGLBVisualAdapter(self.fastglb_mesh)
        return self._visual
    
    def export(self, path):
        return self.fastglb_mesh.export(path)
    
    def apply_transform(self, matrix):
        transformed_fastglb = self.fastglb_mesh.apply_transform(matrix)
        return FastGLBToTrimeshAdapter(transformed_fastglb)


class FastGLBVisualAdapter:
    """Адаптер для visual свойств"""
    
    def __init__(self, fastglb_mesh):
        self.fastglb_mesh = fastglb_mesh
        self._material = None
    
    @property
    def kind(self):
        return self.fastglb_mesh.visual.kind
    
    @property
    def vertex_colors(self):
        return self.fastglb_mesh.visual.vertex_colors
    
    @property
    def uv(self):
        return self.fastglb_mesh.visual.uv
    
    @property
    def material(self):
        if self._material is None:
            self._material = FastGLBMaterialAdapter(self.fastglb_mesh)
        return self._material


class FastGLBMaterialAdapter:
    """Адаптер для материалов"""
    
    def __init__(self, fastglb_mesh):
        self.fastglb_mesh = fastglb_mesh
    
    @property
    def baseColorTexture(self):
        return self.fastglb_mesh.textures.get('albedo')
    
    @property
    def metallicRoughnessTexture(self):
        return self.fastglb_mesh.textures.get('metallicRoughness')
    
    def to_pbr(self):
        return self


def unpatch_trimesh_load():
    """Восстанавливает оригинальный trimesh.load"""
    # Эта функция может быть вызвана для отката изменений
    pass


# Автоматический патч при импорте (опционально)
def auto_patch():
    """Автоматически патчит trimesh при импорте модуля"""
    try:
        patch_trimesh_load()
    except ImportError:
        print("⚠️ trimesh не найден, патч не применен")
    except Exception as e:
        print(f"⚠️ Ошибка применения патча: {e}")


# Простая функция для замены в существующем коде
def load_with_fastglb(path):
    """Простая замена для trimesh.load в существующем коде"""
    
    if path.lower().endswith(('.glb', '.gltf')):
        print(f"🚀 Загрузка через FastGLB: {path}")
        fastglb_mesh = fastglb.load(path)
        return FastGLBToTrimeshAdapter(fastglb_mesh)
    else:
        # Fallback к trimesh для других форматов
        import trimesh
        return trimesh.load(path)


def fastglb_to_comfy_mesh(fastglb_mesh) -> 'Mesh':
    """Конвертирует FastGLB в Comfy Mesh объект"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mesh = Mesh()
    mesh.device = device
    
    # Основная геометрия
    if len(fastglb_mesh.vertices) > 0:
        mesh.v = torch.tensor(fastglb_mesh.vertices, dtype=torch.float32, device=device)
        mesh.f = torch.tensor(fastglb_mesh.faces, dtype=torch.int32, device=device)
    else:
        return None
    
    # UV координаты
    if fastglb_mesh.uv_coordinates is not None:
        mesh.vt = torch.tensor(fastglb_mesh.uv_coordinates, dtype=torch.float32, device=device)
        mesh.ft = mesh.f
    
    # Нормали
    if fastglb_mesh.vertex_normals is not None:
        mesh.vn = torch.tensor(fastglb_mesh.vertex_normals, dtype=torch.float32, device=device)
        mesh.fn = mesh.f
    
    # Цвета вершин
    if fastglb_mesh.vertex_colors is not None:
        vertex_colors_float = fastglb_mesh.vertex_colors[:, :3].astype(np.float32) / 255.0
        mesh.vc = torch.tensor(vertex_colors_float, dtype=torch.float32, device=device)
    
    # Текстуры
    if 'albedo' in fastglb_mesh.textures and fastglb_mesh.textures['albedo'] is not None:
        albedo_float = fastglb_mesh.textures['albedo'].astype(np.float32) / 255.0
        mesh.albedo = torch.tensor(albedo_float, dtype=torch.float32, device=device).contiguous()
    else:
        mesh.set_new_albedo(1024, 1024)
    
    if 'metallicRoughness' in fastglb_mesh.textures and fastglb_mesh.textures['metallicRoughness'] is not None:
        mr_float = fastglb_mesh.textures['metallicRoughness'].astype(np.float32) / 255.0
        mesh.metallicRoughness = torch.tensor(mr_float, dtype=torch.float32, device=device).contiguous()
    
    return mesh