"""
ПОЛНАЯ замена trimesh.load БЕЗ зависимости от trimesh
"""

import os
from .mesh import Mesh, FastMesh
from .fastglb import FastGLB, FastGLBLoader


def load_mesh_without_trimesh(path, use_fastmesh=True, **kwargs):
    """
    Полная замена trimesh.load() БЕЗ использования trimesh
    
    Args:
        path: путь к файлу
        use_fastmesh: использовать FastMesh для GLB/GLTF
        **kwargs: дополнительные параметры
    
    Returns:
        Mesh или FastMesh объект
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    file_ext = os.path.splitext(path)[1].lower()
    
    try:
        if file_ext in ['.glb', '.gltf']:
            print(f"🚀 Загружаем GLB/GLTF БЕЗ trimesh: {path}")
            
            if use_fastmesh:
                # Используем FastMesh
                mesh = FastMesh.load(path, **kwargs)
                return _convert_to_trimesh_like(mesh) if mesh else None
            else:
                # Используем стандартный Mesh с нашим GLTF загрузчиком
                mesh = Mesh.load_gltf(path, **kwargs)
                return _convert_to_trimesh_like(mesh) if mesh else None
                
        elif file_ext == '.obj':
            print(f"🔧 Загружаем OBJ БЕЗ trimesh: {path}")
            
            if use_fastmesh:
                mesh = FastMesh.load(path, **kwargs)
            else:
                mesh = Mesh.load_obj(path, **kwargs)
                
            return _convert_to_trimesh_like(mesh) if mesh else None
            
        elif file_ext == '.ply':
            print(f"⚠️ PLY поддержка ограничена, fallback к базовой загрузке")
            # Для PLY пока используем простую загрузку
            mesh = _load_ply_simple(path, **kwargs)
            return _convert_to_trimesh_like(mesh) if mesh else None
            
        else:
            raise NotImplementedError(f"Формат {file_ext} не поддерживается mesh_processor")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки {path}: {e}")
        raise


def _convert_to_trimesh_like(mesh):
    """Конвертирует Mesh/FastMesh в объект с trimesh-подобным API"""
    
    class TrimeshCompatible:
        def __init__(self, mesh_obj):
            self._mesh = mesh_obj
            
        @property
        def vertices(self):
            return self._mesh.v.detach().cpu().numpy() if self._mesh.v is not None else None
            
        @property
        def faces(self):
            return self._mesh.f.detach().cpu().numpy() if self._mesh.f is not None else None
            
        @property
        def vertex_normals(self):
            if self._mesh.vn is None:
                self._mesh.auto_normal()
            return self._mesh.vn.detach().cpu().numpy() if self._mesh.vn is not None else None
            
        @property
        def visual(self):
            return TrimeshVisualCompatible(self._mesh)
            
        def export(self, path):
            return self._mesh.write(path)
            
        def apply_transform(self, matrix):
            # Создаем копию и применяем трансформацию
            import torch
            import numpy as np
            
            # Копируем mesh
            if isinstance(self._mesh, FastMesh):
                new_mesh = FastMesh(
                    v=self._mesh.v.clone(),
                    f=self._mesh.f.clone(),
                    vn=self._mesh.vn.clone() if self._mesh.vn is not None else None,
                    fn=self._mesh.fn.clone() if self._mesh.fn is not None else None,
                    vt=self._mesh.vt.clone() if self._mesh.vt is not None else None,
                    ft=self._mesh.ft.clone() if self._mesh.ft is not None else None,
                    vc=self._mesh.vc.clone() if self._mesh.vc is not None else None,
                    albedo=self._mesh.albedo.clone() if self._mesh.albedo is not None else None,
                    metallicRoughness=self._mesh.metallicRoughness.clone() if self._mesh.metallicRoughness is not None else None,
                    device=self._mesh.device
                )
            else:
                new_mesh = Mesh(
                    v=self._mesh.v.clone(),
                    f=self._mesh.f.clone(),
                    vn=self._mesh.vn.clone() if self._mesh.vn is not None else None,
                    fn=self._mesh.fn.clone() if self._mesh.fn is not None else None,
                    vt=self._mesh.vt.clone() if self._mesh.vt is not None else None,
                    ft=self._mesh.ft.clone() if self._mesh.ft is not None else None,
                    vc=self._mesh.vc.clone() if self._mesh.vc is not None else None,
                    albedo=self._mesh.albedo.clone() if self._mesh.albedo is not None else None,
                    metallicRoughness=self._mesh.metallicRoughness.clone() if self._mesh.metallicRoughness is not None else None,
                    device=self._mesh.device
                )
            
            # Применяем трансформацию к вершинам
            if new_mesh.v is not None:
                matrix_torch = torch.tensor(matrix, dtype=torch.float32, device=new_mesh.device)
                v_homogeneous = torch.cat([new_mesh.v, torch.ones(new_mesh.v.shape[0], 1, device=new_mesh.device)], dim=1)
                v_transformed = (matrix_torch @ v_homogeneous.T).T[:, :3]
                new_mesh.v = v_transformed
            
            # Применяем к нормалям (только поворот)
            if new_mesh.vn is not None:
                rotation_matrix = torch.tensor(matrix[:3, :3], dtype=torch.float32, device=new_mesh.device)
                new_mesh.vn = (rotation_matrix @ new_mesh.vn.T).T
            
            return TrimeshCompatible(new_mesh)
            
        def copy(self):
            return TrimeshCompatible(self._mesh)
    
    return TrimeshCompatible(mesh)


class TrimeshVisualCompatible:
    """Совместимый visual объект"""
    
    def __init__(self, mesh_obj):
        self._mesh = mesh_obj
        
    @property
    def kind(self):
        if self._mesh.vc is not None:
            return 'vertex'
        elif self._mesh.albedo is not None:
            return 'texture'
        else:
            return 'vertex'
            
    @property
    def vertex_colors(self):
        if self._mesh.vc is not None:
            colors = self._mesh.vc.detach().cpu().numpy()
            import numpy as np
            alpha = np.ones((colors.shape[0], 1))
            return (np.concatenate([colors, alpha], axis=1) * 255).astype(np.uint8)
        else:
            import numpy as np
            num_vertices = self._mesh.v.shape[0] if self._mesh.v is not None else 0
            return np.full((num_vertices, 4), 255, dtype=np.uint8)
            
    @property
    def uv(self):
        if self._mesh.vt is not None:
            uv = self._mesh.vt.detach().cpu().numpy()
            uv[:, 1] = 1.0 - uv[:, 1]  # Переворачиваем обратно
            return uv
        else:
            return None
            
    @property 
    def material(self):
        return TrimeshMaterialCompatible(self._mesh)


class TrimeshMaterialCompatible:
    """Совместимый material объект"""
    
    def __init__(self, mesh_obj):
        self._mesh = mesh_obj
        
    @property
    def baseColorTexture(self):
        if self._mesh.albedo is not None:
            import numpy as np
            texture = self._mesh.albedo.detach().cpu().numpy()
            return (texture * 255).astype(np.uint8)
        else:
            return None
            
    @property
    def metallicRoughnessTexture(self):
        if self._mesh.metallicRoughness is not None:
            import numpy as np
            texture = self._mesh.metallicRoughness.detach().cpu().numpy()
            return (texture * 255).astype(np.uint8)
        else:
            return None
            
    def to_pbr(self):
        return self


def _load_ply_simple(path, device=None, **kwargs):
    """Простая загрузка PLY без trimesh"""
    
    import torch
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        vertices = []
        faces = []
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Парсим PLY header
        vertex_count = 0
        face_count = 0
        in_header = True
        vertex_section = False
        face_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])
                elif line == 'end_header':
                    in_header = False
                    vertex_section = True
                    continue
            
            elif vertex_section:
                if len(vertices) < vertex_count:
                    coords = list(map(float, line.split()[:3]))
                    vertices.append(coords)
                else:
                    vertex_section = False
                    face_section = True
            
            if face_section and len(faces) < face_count:
                parts = line.split()
                if len(parts) >= 4:  # "3 v1 v2 v3"
                    face = [int(parts[1]), int(parts[2]), int(parts[3])]
                    faces.append(face)
        
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("Не удалось прочитать PLY файл")
        
        # Создаем Mesh
        mesh = Mesh(
            v=torch.tensor(vertices, dtype=torch.float32, device=device),
            f=torch.tensor(faces, dtype=torch.int32, device=device),
            device=device
        )
        
        print(f"✅ PLY загружен: {len(vertices)} вершин, {len(faces)} граней")
        return mesh
        
    except Exception as e:
        print(f"❌ Ошибка загрузки PLY: {e}")
        return None


# Главная функция замены
def replace_trimesh_load():
    """
    Заменяет trimesh.load на нашу функцию
    
    Использование:
        from mesh_processor.trimesh_replacement import replace_trimesh_load
        replace_trimesh_load()
        
        # Теперь можно использовать как обычно:
        mesh = trimesh.load("model.glb")  # Будет использовать нашу функцию!
    """
    
    try:
        import trimesh
        # Сохраняем оригинальную функцию
        trimesh._original_load = trimesh.load
        
        # Заменяем на нашу
        trimesh.load = load_mesh_without_trimesh
        
        print("✅ trimesh.load заменен на mesh_processor версию!")
        
    except ImportError:
        print("⚠️ trimesh не найден, замена не нужна")


def restore_trimesh_load():
    """Восстанавливает оригинальный trimesh.load"""
    
    try:
        import trimesh
        if hasattr(trimesh, '_original_load'):
            trimesh.load = trimesh._original_load
            delattr(trimesh, '_original_load')
            print("✅ Оригинальный trimesh.load восстановлен")
        else:
            print("⚠️ Оригинальный trimesh.load не найден")
    except ImportError:
        print("⚠️ trimesh не найден")


# Экспорт функций
__all__ = [
    'load_mesh_without_trimesh',
    'replace_trimesh_load', 
    'restore_trimesh_load'
]
