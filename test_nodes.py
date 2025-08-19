import os
import torch

class Load_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, resize, renormal, retex, optimizable, clean, resize_bound):
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if os.path.exists(mesh_file_path):
            folder, filename = os.path.split(mesh_file_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(not optimizable):
                    mesh = Mesh.load(mesh_file_path, resize, renormal, retex, clean, resize_bound)
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {mesh_file_path} does not exist").error.print()
        return (mesh, )


class Load_3D_Mesh_New:
    """Нода для тестирования нового GLTF загрузчика на базе mesh_processor"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "use_new_gltf_loader": ("BOOLEAN", {"default": True, "tooltip": "Использовать новый mesh_processor загрузчик для GLB/GLTF"}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, use_new_gltf_loader, resize, renormal, retex, optimizable, clean, resize_bound):
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if os.path.exists(mesh_file_path):
            folder, filename = os.path.split(mesh_file_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(not optimizable):
                    mesh = Mesh.load(
                        mesh_file_path, 
                        resize=resize, 
                        renormal=renormal, 
                        retex=retex, 
                        clean=clean, 
                        bound=resize_bound,
                        use_new_gltf_loader=use_new_gltf_loader
                    )
                    
                    # Логируем какой загрузчик использовался
                    if filename.lower().endswith(('.glb', '.gltf')):
                        loader_type = "новый mesh_processor" if use_new_gltf_loader else "старый trimesh"
                        print(f"[Load_3D_Mesh_New] Использован {loader_type} загрузчик для {filename}")
                        
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {mesh_file_path} does not exist").error.print()
        return (mesh, )


class Load_3D_Mesh_Compare:
    """Нода для сравнения старого и нового загрузчиков GLTF"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
                "clean": ("BOOLEAN", {"default": False},),
                "resize_bound": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
        "MESH", 
        "STRING",
    )
    RETURN_NAMES = (
        "mesh_old",
        "mesh_new",
        "comparison_report",
    )
    FUNCTION = "load_and_compare"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_and_compare(self, mesh_file_path, resize, renormal, retex, optimizable, clean, resize_bound):
        import time
        
        mesh_old = None
        mesh_new = None
        report = ""
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if not os.path.exists(mesh_file_path):
            report = f"❌ Файл не найден: {mesh_file_path}"
            return (mesh_old, mesh_new, report)
            
        folder, filename = os.path.split(mesh_file_path)
        if not filename.lower().endswith(('.glb', '.gltf')):
            report = f"❌ Файл {filename} не является GLB/GLTF"
            return (mesh_old, mesh_new, report)
        
        with torch.inference_mode(not optimizable):
            # Загрузка старым способом
            try:
                start_time = time.time()
                mesh_old = Mesh.load(
                    mesh_file_path, resize, renormal, retex, clean, resize_bound,
                    use_new_gltf_loader=False
                )
                old_time = time.time() - start_time
                old_status = f"✅ {old_time:.3f}s"
            except Exception as e:
                old_status = f"❌ {str(e)[:50]}"
                old_time = 0
            
            # Загрузка новым способом
            try:
                start_time = time.time()
                mesh_new = Mesh.load(
                    mesh_file_path, resize, renormal, retex, clean, resize_bound,
                    use_new_gltf_loader=True
                )
                new_time = time.time() - start_time
                new_status = f"✅ {new_time:.3f}s"
            except Exception as e:
                new_status = f"❌ {str(e)[:50]}"
                new_time = 0
            
            # Создаем отчет
            report_lines = [
                f"📊 Сравнение загрузчиков для {filename}",
                f"🕰️ Trimesh: {old_status}",
                f"🆕 MeshProcessor: {new_status}",
            ]
            
            if mesh_old and mesh_new and old_time > 0 and new_time > 0:
                speed_diff = ((new_time / old_time - 1) * 100)
                report_lines.append(f"⚡ Скорость: {speed_diff:+.1f}%")
                
                # Сравнение геометрии
                if mesh_old.v is not None and mesh_new.v is not None:
                    vertices_match = torch.allclose(mesh_old.v, mesh_new.v, atol=1e-5)
                    report_lines.append(f"📐 Вершины: {'✅' if vertices_match else '❌'}")
                
                if mesh_old.f is not None and mesh_new.f is not None:
                    faces_match = torch.equal(mesh_old.f, mesh_new.f)
                    report_lines.append(f"🔺 Грани: {'✅' if faces_match else '❌'}")
                
                # UV и текстуры
                uv_old = mesh_old.vt is not None
                uv_new = mesh_new.vt is not None
                report_lines.append(f"🗺️ UV: старый={uv_old}, новый={uv_new}")
                
                tex_old = mesh_old.albedo is not None
                tex_new = mesh_new.albedo is not None
                report_lines.append(f"🎨 Текстуры: старый={tex_old}, новый={tex_new}")
            
            report = "\n".join(report_lines)
        
        return (mesh_old, mesh_new, report)